#pragma once

// Device backend for the fused equilibrium-Xe + ReconstructNode node loop
// (XsReconKernel.h), behind RASBERY_GPU_XSRECON (default off).
//
// Placement notes, in the style of CudaDepletionBackend.h:
//
//  - One backend instance per XSSet (per Driver), each with its own stream, so
//    M concurrent Drivers overlap on the device without a rendezvous.  The
//    CMFD arena's slot rendezvous is NOT reused here for the same reason the
//    depletion probe declined it: equilibrium-Xe calls are outer-loop events
//    that drift apart across batch instances, so a rendezvous would starve.
//
//  - The per-instance _micx block (~30 MB at APR1400 size) is device-resident
//    between calls and re-uploaded only when the host-side generation counter
//    says the host mutated it (UpdateFlatXS / Update / rod cusping).  Between
//    those mutations a whole Xe<->flux cascade runs against the resident copy.
//    Uploading it unconditionally per call is the design the depletion probe's
//    Sec 5.3 arithmetic pre-rejects.
//
//  - Failure policy follows the depletion probe, not the CMFD backend: any
//    CUDA failure makes solve() return false and the caller falls back to the
//    bit-identical CPU loop, with one stderr warning.  This is an
//    off-by-default exploration path; fail-open-to-CPU keeps physics runs
//    alive on machines without a device.

#include "GpuCanonicalState.h"
#include "HostPinRegistry.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

namespace rasbery {

namespace xsrecon {
struct BatchView;
}

namespace xe {
struct XeTriple;
struct XeTxnControl;
}

namespace flatxs {
struct FlatXsView;
}

namespace nodal {
/// WP20.1: `NodalView` is an ALIAS now (NodalViewT<double>, src/NodalKernel.h),
/// so it cannot be forward-declared as a struct -- the alias and the
/// declaration would conflict at the first TU that sees both.  The template is
/// forward-declared instead and the alias restated, which is the same promise
/// to a caller that only takes a reference.
template <class ValueT>
struct NodalViewT;
using NodalView = NodalViewT<double>;
}

/// Element counts of the flat coefficient tables, so the backend can size the
/// shared device copy without seeing XSSet.  All counts are in doubles except
/// n_deltas/n_knots.
struct FlatXsLibShape {
    std::size_t lmp_slot;  ///< doubles per coeff_lmp[t] array
    std::size_t lsm;       ///< doubles in coeff_lsm
    std::size_t mic_slot;  ///< doubles per coeff_mic[t] array (0 without micx)
    std::size_t msm;       ///< doubles in coeff_msm (0 without micx)
    std::size_t n_knots;
    std::size_t n_deltas;
};

class XsReconBackend {
public:
    XsReconBackend();
    ~XsReconBackend();

    XsReconBackend(const XsReconBackend&)            = delete;
    XsReconBackend& operator=(const XsReconBackend&) = delete;

    /// True when a device was found and every allocation so far has succeeded.
    bool available() const;

    /// Human-readable reason when available() is false.
    const std::string& status() const;

    /// Run one fused equilibrium-Xe step for every fuel node in `host`
    /// (host-side pointers).  Uploads phif/iden/lmp/xs every call and mic only
    /// when `micx_generation` differs from the resident copy's generation.
    /// On success writes the raw max relative Xe step to *max_change_out,
    /// downloads xs (all slots + scatter) and the three Xe-chain iden rows
    /// back into the host arrays, and returns true.  On any CUDA error returns
    /// false with the host arrays untouched beyond what a partial upload could
    /// never touch (uploads copy host->device only).
    /// state_generation tracks host-side writes to _xs/_iden outside the
    /// backend downloads; while it matches the resident copy, the per-call
    /// xs/iden uploads are skipped (host and device are bit-identical).
    bool solve(const xsrecon::BatchView& host, unsigned long long micx_generation,
               unsigned long long state_generation, double* max_change_out);

    // --- Rev.7.1 Task 13: the SPLIT Xe arm (RASBERY_GPU_XE) ----------------
    //
    // solve() above is the FUSED step -- evaluate, damp, write, reconstruct --
    // and it is the whole of XSSet::UpdateEquilibriumXenon.  The safeguarded
    // Anderson arm cannot use it: it has to see F(x) before it commits
    // anything.  These six take the same node body apart at the same seam the
    // host API is split at (SnapshotXenon / EvaluateEquilibriumXenon /
    // CommitXenon) and keep the Anderson history DEVICE-RESIDENT between them,
    // which is the point -- ten triples of 3*n_fuel doubles never cross the bus.
    //
    // ONE HISTORY PER BACKEND, AND THEREFORE PER DRIVER.  A backend belongs to
    // one XSSet, an XSSet to one Driver, and a --batch-mode deck to one Driver,
    // so a batch of M decks has M independent histories with no arrangement
    // needed.  That is deliberate and it is the fix for a bug this tree has
    // already had: a process-wide slot-0 buffer that every Driver adopted.
    // Nothing here is static.
    //
    // Every one returns false on any unavailability or CUDA error, having
    // written nothing to the host, and the caller then runs the untouched host
    // path.  The state uploads follow solve()'s residency contract exactly.

    /// Evaluate the map at the current inventory: x (the snapshot), F(x) and
    /// g = F(x) - x into the device history, and the RAW maximum relative
    /// Xe-135 step into *picard_out.  WRITES NOTHING to the host arrays --
    /// no iden row, no xs entry -- so a caller that rejects the image leaves
    /// the solver exactly as it found it.
    bool xeEvaluate(const xsrecon::BatchView& host, unsigned long long micx_generation,
                    unsigned long long state_generation, double* picard_out);

    /// df[0] <- df[1], dg[0] <- dg[1]: the oldest window column falls out.
    bool xeRotateHistory();

    /// df[col] <- f - f_prev, dg[col] <- g - g_prev.
    bool xeRecordColumn(int col);

    /// f_prev <- f, g_prev <- g.
    bool xeSaveEvaluation();

    /// The six inner products of the depth-2 normal equations, in
    /// xe::XeDotSlot order.  `ncol` selects which of them are actually
    /// computed: a one-column window reads only <g,g>, <dg0,dg0> and <dg0,g>,
    /// and the unread slots are left at zero rather than filled from a buffer
    /// no evaluation has written.
    ///
    /// FIXED PARTITION, and that is the N1 line: the host folds ~3*n_fuel
    /// terms into one running sum, this cuts the range into a partition count
    /// that depends on nothing but a constant, so it is reproducible run to run
    /// but associates the additions differently.  See XeKernel.h.
    bool xeDots(int ncol, double* out_six);

    /// cand <- F - sum_j gamma_j dF_j, plus the two numbers the safeguards
    /// need: *step_out is the trust-region metric and *physics_ok is false when
    /// any component came out non-finite or negative.
    bool xeCandidate(const double* gamma, int ncol, double* step_out, bool* physics_ok);

    /// Commit an image and reconstruct.  `triple` is an xe::XeTripleId --
    /// XE_T_CAND for an accepted Anderson candidate, and the Picard path passes
    /// XE_T_F with `relax` to commit x + relax*(F - x) instead.
    ///
    /// `picard_skip` reproduces the FUSED update's skip: that path neither
    /// writes nor reconstructs a node whose normalized flux was not positive,
    /// where CommitXenon reconstructs every fuel node it is given.  Two
    /// different host functions with two different contracts, and the device
    /// has to honour both.
    ///
    /// Downloads xs (all slots + scatter) and the three Xe-chain iden rows into
    /// the host arrays, so the host is authoritative again on return and the
    /// caller must NOT bump its host-state generation -- the same contract
    /// solve() has.
    bool xeCommit(const xsrecon::BatchView& host, int triple, double relax,
                  bool picard_skip, unsigned long long state_generation);

    /// Receipts (process-wide, all instances): fuel-node evaluations and
    /// commits the device actually ran.  Zero means the arm never fired,
    /// whatever the flag said -- the G0 validity check.
    // --- WP7 stage C: the whole step as ONE device transaction -------------
    //
    // Everything the host decides between the evaluate and the commit, decided
    // on the device instead: a step becomes evaluate -> history -> dots ->
    // solve/accept -> commit with ONE host synchronisation, the drain's, which
    // the caller was paying for anyway.  The six entry points above stay and
    // are what RASBERY_GPU_XE_TXN=0 runs, so the bit-identity gate compares
    // against live code rather than a memory of it.
    //
    // The request carries what the host still owns and the device cannot know:
    // the window bookkeeping (Driver.h's XeAndersonState is host state), and
    // the three constants Driver.h declares.  Passing the constants rather than
    // redeclaring them is the point -- a second spelling of
    // XE_ANDERSON_MIN_GRAM is a second opinion waiting to drift.
    struct XeTxnRequest {
        int    hist_col    = -1;   ///< window column to record; -1 records none
        bool   hist_rotate = false;///< drop the oldest column before recording
        int    ncol        = 0;    ///< window width AFTER the record; 0 = unarmed
        double eq_tol      = 0.0;  ///< Driver.h XE_EQUILIBRIUM_TOLERANCE
        double min_gram    = 0.0;  ///< Driver.h XE_ANDERSON_MIN_GRAM
        double max_step    = 0.0;  ///< the trust region, as a multiple of picard
        double relax       = 1.0;  ///< the damping a REJECTED step commits with
    };

    /// Run one Xe step as a single device transaction.  Returns false having
    /// written nothing observable, in which case the caller must run the
    /// round-tripping arm; on true `*out` carries the step's outcome, residual
    /// and reason, and the host arrays are authoritative again.
    bool xeTransaction(const xsrecon::BatchView& host,
                       unsigned long long micx_generation,
                       unsigned long long state_generation, const XeTxnRequest& req,
                       xe::XeTxnControl* out);

    static unsigned long long xeEvaluations();
    static unsigned long long xeCommits();

    /// Run one unrodded flat-XS update for every node in `host` (host-side
    /// pointers; see flatxs::FlatXsView for the stream contract).
    ///
    ///  - The library coefficient tables are uploaded once per process per
    ///    distinct content (shared across instances, keyed by a full FNV over
    ///    the bytes; `shape` sizes the copy).
    ///  - The reference blocks re-upload only when `ref_generation` differs
    ///    from the resident copy's.
    ///  - The live micx/lmpx blocks (shared storage with the xsrecon solve)
    ///    re-upload only when `micx_generation` differs from the resident
    ///    generation; the kernel then rewrites the target columns, the result
    ///    downloads whole, and on `mark_micx_resident` the resident generation
    ///    advances to `micx_generation_next` so the next xsrecon call skips
    ///    its ~70 MB re-upload.  Callers pass mark_micx_resident=false when
    ///    rodded nodes will mutate the host arrays right after this call.
    ///  - xs (all slots + scatter) and iden upload whole per call so the
    ///    kernel's target-only writes round-trip every other column unchanged.
    ///
    /// On success downloads lmp/mic (active slots + scatter), xs, and the
    /// three light-isotope iden rows back into the host arrays and returns
    /// true.  On any CUDA error returns false; uploads only copy host->device,
    /// so a failed call leaves the host arrays untouched.
    bool solveFlatXs(const flatxs::FlatXsView& host, const FlatXsLibShape& shape,
                     unsigned long long micx_generation,
                     unsigned long long micx_generation_next,
                     unsigned long long ref_generation,
                     unsigned long long state_generation, bool mark_micx_resident);

    // --- WP15: the deferred micx/lmpx download (RASBERY_GPU_MICX_RESIDENT) --
    //
    // With the arm on, solveFlatXs computes the micx/lmpx block and LEAVES IT
    // ON THE DEVICE -- 59.5 MB of its 61.5 MB per-call download, and 85 % of the
    // run's whole D2H side (docs/WP13 §4.4).  The download is not skipped, it is
    // OWED: these three say what is owed and pay it.
    //
    // THE CONTRACT THE CALLER MUST KEEP.  While either flag is true no host code
    // may read or write the arrays the view points at.  XSSet::EnsureMicxHost is
    // the single payer and every host reader goes through it; solveFlatXs and
    // stage() carry a loud guard for the case where one does not.
    //
    // THE TWO HALVES ARE TRACKED APART because they have different readers: the
    // depletion / Xe / CRAM consumers read the eleven scalar slots and none of
    // them reads the scatter block, so `downloadFlatXsMicx(v, true, false)` is a
    // 49.0 MB answer that is complete for them.

    /// True when the 9 mic + 9 lmp active slots live only on the device.
    [[nodiscard]] bool micxScalarsPending() const;

    /// True when msm + lsm live only on the device.
    [[nodiscard]] bool micxScatterPending() const;

    /// The micx/lmpx generation the resident device block holds (0 = none).
    [[nodiscard]] unsigned long long micxResidentGeneration() const;

    // --- WP15.1: handing the resident block to another backend -------------
    //
    // The CRAM depletion backend uploads four of these eleven blocks from the
    // HOST on every depletion step (CudaCramBackend.cu, `H2D mic`), and the
    // host copy it reads came out of THIS device block at the last flat-XS
    // solve.  device -> host -> device, 21 MB each way per step.  These two let
    // it read the block where it already is.
    //
    // THE CALLER MUST CHECK THE GENERATION.  `micxDeviceSlot` hands out an
    // address, not a promise about WHICH epoch is in it; a consumer keyed on
    // XSSet::_micx_generation must compare that against
    // micxResidentGeneration() and fall back to its H2D when they differ.

    /// Device address of one of the eleven micro-XS slots inside the resident
    /// block, or nullptr when nothing is resident (or the backend is a stub).
    ///
    /// WP20.1: `const void*`, not `const double*`.  Under RASBERY_GPU_FP32 the
    /// block is float, and a typed pointer would let a consumer copy
    /// `count * sizeof(double)` bytes out of it -- half of them the next
    /// slot's.  Ask micxDeviceElemBytes() and either memcpy or widen.
    [[nodiscard]] const void* micxDeviceSlot(int xt) const;

    /// 8 on the FP64 arm, 4 under RASBERY_GPU_FP32.  A consumer that ignores
    /// this and assumes 8 reads twice the block it was given.
    [[nodiscard]] int micxDeviceElemBytes() const;

    /// A cudaEvent_t (as void*) recorded NOW on this backend's stream, which a
    /// consumer's stream must wait on before reading what micxDeviceSlot
    /// returned.  nullptr means no ordering handover is available and the
    /// consumer must not do the D2D.
    void* micxReadyEvent();

    /// WP15.1 receipts for the batch nodal arena's jnet upload shadow.
    /// `tests` is the denominator: 0 tests means the arm was never consulted,
    /// which reads very differently from consulted-and-always-missed.
    static unsigned long long nodalJnetElidedBytes();
    static unsigned long long nodalJnetElisionHits();
    static unsigned long long nodalJnetElisionTests();

    /// Issue the copies solveFlatXs deferred, into `host`'s live micx/lmpx
    /// pointers, and DRAIN -- the caller is about to dereference them.  The
    /// same copy list off the same offsets solveFlatXs would have used, so a
    /// materialised array is bit-for-bit the eagerly downloaded one.
    /// Idempotent: with nothing owed (or nothing requested) it returns true
    /// having moved no bytes and taken no synchronisation.
    bool downloadFlatXsMicx(const flatxs::FlatXsView& host, bool scalars, bool scatter);

    /// WP15 receipts, process-wide over every instance.  `micxResidentHits` is
    /// the G0 validity check: 0 with the arm on means the flag never reached a
    /// solve and any saving quoted from this run is void.
    static unsigned long long micxResidentHits();
    static unsigned long long micxLazyDownloads();
    static unsigned long long micxSliceDownloads();
    static unsigned long long micxBytesSaved();

    /// WP15 §2 census: nodal coefficient H2D copies and their bytes.  Nothing
    /// elides them (see the note at the upload site); they are counted so the
    /// claim that `_const_generation` advances once per Xe device step can be
    /// checked rather than argued.
    static unsigned long long nodalConstUploads();
    static unsigned long long nodalConstBytes();

    /// Total fuel nodes processed on the device by this process (all
    /// instances).  Zero means the device path never ran, whatever the flag
    /// said -- the G0 validity receipt.
    static unsigned long long nodesSolved();

    /// Same receipt for the flat-XS kernel (RASBERY_GPU_FLATXS).
    static unsigned long long flatXsNodesSolved();

    /// Run one nodal drive() (the five per-outer phases) on the device.
    /// Geometry tables upload once; the nine updateConstant arrays re-upload
    /// on `const_generation`; chif on `ref_generation`; the working arrays
    /// are device-only.  xsrf/xsnf/xssm are read from the resident xs block
    /// when `state_generation` matches, else uploaded for this call (without
    /// advancing the residency -- iden may still be stale).  Per call: jnet
    /// and flux upload, jnet and phis download.  Fail-open to the CPU body.
    bool solveNodal(const nodal::NodalView& host,
                    unsigned long long const_generation,
                    unsigned long long ref_generation,
                    unsigned long long state_generation);

    /// Hybrid tail: after the caller ran the host calculateEven on the
    /// arrays solveNodal downloaded, upload dsncff and finish with the jnet
    /// phase.  Only valid right after a hybrid solveNodal returned true.
    bool solveNodalPost(const nodal::NodalView& host);

    // --- Rev.7.1 Task 7: canonical CMFD-Nodal device state -----------------
    //
    // EXTERNAL-BUFFER ADAPTER MODE.  The caller hands over device pointers this
    // backend must USE instead of its own private block for jnet/flux/phis.
    // They are BORROWED, never freed here, and they must come from
    // GpuPhysicsArena -- whose whole contract is that a per-slot address is
    // fixed at reserve() and never moves, which is what makes it legal to bake
    // them into the captured nodal graph.
    //
    // Passing an all-null set (the default) is LEGACY: the private block is
    // used, every transfer happens exactly as before, and the feature-off path
    // is byte-identical.  That is also what makes mixed mode work -- one
    // instance shared, another legacy, in one process, with no third code path.
    void adoptCanonicalBuffers(const gpu::CanonicalSlotBuffers& buffers);

    /// What this backend is currently borrowing (all-null in legacy mode).
    [[nodiscard]] gpu::CanonicalSlotBuffers canonicalBuffers() const;

    /// Rev.7.1 Task 18 RETIRED THE `IS IT HONOURED` PREDICATE.
    ///
    /// It answered `does the drive that consumes an adopted set actually read
    /// it, or does it upload over it` -- and the only reason the question
    /// existed is that the BATCHED nodal arena did the latter.  The arena now
    /// addresses jnet/flux/phis through its view table and consults the same
    /// elision predicate the per-instance arm does (CudaXsReconBackend.cu,
    /// launchBatch), so both arms honour the binding and the device outer
    /// segment can drop its jnet bridge on every arm rather than only the one.

    /// THE OBSERVATION API (Rev.7 Sec 3.3).  With the routine downloads elided,
    /// the host Geometry arrays no longer track the device -- so a host consumer
    /// must SAY it is about to look.  `mask` is a bitmask over CanonicalRegion
    /// (canonicalConsumerMask() builds the right one per consumer); the regions
    /// in it are copied back on the next drive, the rest are not.
    ///
    /// Set it to 0 again once the consumer has run, or every drive pays the
    /// download the sharing was supposed to remove.
    void setMaterializeMask(std::uint32_t mask);
    [[nodiscard]] std::uint32_t materializeMask() const;

    /// Rev.7.1 Task 18-lite: the device outer segment's per-drive declaration.
    ///
    /// THE OBSERVATION API IS AN AWKWARD FIT FOR A SEGMENT, and this is the
    /// shape that fits.  The mask alone says which regions come BACK; it says
    /// nothing about which ones need to go OUT, and inside a segment that is the
    /// question that matters -- the device jnet is the only current copy, so an
    /// upload of the host's would destroy the outer.  One call sets both halves
    /// so they cannot disagree:
    ///
    ///   in_segment = true   jnet/phis device-owned, flux device-owned iff the
    ///                       drive left it there; nothing materialised, because
    ///                       the segment mirrors both arrays itself at its exit.
    ///   in_segment = false  every region host-owned and jnet/phis materialised:
    ///                       exactly the transfers a drive made before Task 7,
    ///                       which is what a host outer body and every non-
    ///                       segment consumer still expect.
    ///
    /// A NO-OP ON A LEGACY INSTANCE (no canonical set adopted), so a call site
    /// does not have to ask whether the adoption happened.  Idempotent, and it
    /// writes the same values on every outer of a segment, which is what keeps
    /// it from churning the captured nodal graph.
    void setCanonicalNodalSegmentMode(bool in_segment, bool device_owns_flux);

    /// Rev.7.1 W3 item 2: the event that says the LAST solveNodal's own stream
    /// has finished, or nullptr when that drive already drained itself.
    ///
    /// WHY IT IS A HANDOVER AND NOT A PROMISE.  solveNodal ends by ordering its
    /// stream against whoever reads jnet next, and there are exactly two ways to
    /// do that: block the host, or record an event the reader's stream waits on.
    /// It blocks whenever a download landed in a Geometry array (a host reader is
    /// next); it records the event when both downloads were elided, which happens
    /// only inside a device outer segment.  A caller that ignores this pointer
    /// therefore gets the OLD behaviour on every path that needed it and an
    /// UNORDERED read on the path that did not -- so the runner asks for it once
    /// per outer and waits, rather than assuming either answer.
    ///
    /// CONSUME-ONCE, which is why it is not const: the event describes ONE
    /// drive, and reading it is what discharges it.  The next outer's drive may
    /// take the CPU body and enqueue nothing, and it must not inherit this
    /// answer.
    ///
    /// Returned as void* so Driver.h and the runner's hook table stay free of
    /// <cuda_runtime.h>; the one place that casts it back is the runner, which
    /// is a .cu.
    [[nodiscard]] void* nodalCompletionEvent();

    /// Rev.7.1 W3 item 3: the device double the FULL nodal path reads 1/eigv
    /// from, or nullptr when there is no such slot right now.
    ///
    /// WHY THE ADDRESS AND NOT A SETTER.  The reciprocal's journey was device ->
    /// host -> pinned slot -> device: the sweep produced the eigenvalue on the
    /// device, the host read it back so it could divide, and the captured H2D
    /// carried the quotient home.  Inside a device outer segment the first hop
    /// is the round trip the whole task is removing, so the segment writes the
    /// quotient with a one-thread kernel instead -- and to do that it needs the
    /// address, which only this class knows.
    ///
    /// ASKED PER OUTER AND NEVER CACHED.  The nodal device block is freed and
    /// laid out again whenever nsurf changes (a restart, a different geometry),
    /// so an address held across that is a write into a freed allocation.  It is
    /// null before the first drive -- the block is allocated inside solveNodal --
    /// and on the hybrid arm, where updateMatrix reads the by-value scalar and
    /// there is no slot to write.
    [[nodiscard]] void* nodalReigvDeviceSlot() const;

    /// Rev.7.1 W3 item 3: SOMEBODY ELSE IS WRITING THAT SLOT.
    ///
    /// With this true the FULL path does not upload host.reigv at all: the
    /// caller has already put 1/eigv in the device slot, stream-ordered ahead of
    /// the drive, and an upload would overwrite the device's answer with the
    /// host's copy of it.  With it false -- the default, and every path outside a
    /// segment -- the upload happens exactly as it always did.
    ///
    /// IT IS NOT A GRAPH KEY, AND THAT IS DELIBERATE.  The declaration flips
    /// with the drive: an in-segment canonical outer sets it, the Wielandt
    /// warm-up outer three lines later clears it, and a key would drop and
    /// re-instantiate the captured graph at every one of those flips.  So the
    /// upload was moved OUT of the capture instead -- it is issued on the same
    /// stream immediately before the graph launch, which orders it exactly as
    /// being the graph's first node did, and the capture no longer has an
    /// opinion about it.
    void setNodalReigvDeviceResident(bool resident);

    // -----------------------------------------------------------------------
    // Rev.7.1 Task 10 part 3: THE DRIVE THAT REFUSES TO RUN PAST A DECIDED EXIT
    // -----------------------------------------------------------------------
    //
    // A HOST-FREE SEGMENT CANNOT ASK.  It enqueues its whole budget without
    // returning to the host, so the outers past the exit are already submitted
    // when the transition latches.  Every kernel of the CMFD body reads the
    // segment's per-slot halt word and returns; the nodal drive is a HOST call
    // and could not, so its five kernels now read the same word.
    //
    // AND IT IS NOT OPTIONAL FOR THAT ARM.  The drive is not idempotent -- the
    // transverse leakage is built FROM jnet and the last phase WRITES jnet --
    // so an ungated overrun re-solves on its own output.  Four extra drives on
    // a budget-8 segment that exited at outer 3 is a different answer, not a
    // slower one.
    //
    // @param halt device pointer to the segment's per-slot halt table
    ///       (std::uint32_t*), or nullptr to run ungated -- the default, and
    ///       what every arm outside a host-free segment passes.
    /// @param slot which entry of that table describes this deck.
    ///
    /// BOTH ARE GRAPH KEYS.  The pair is baked into the captured nodal graph,
    /// so flipping the gate re-instantiates once and then never again; the
    /// pointer itself is stable for the life of a run.
    void setNodalHaltGate(const void* halt, int slot);

    /// Rev.7.1 Task 10 part 3: THE OTHER HALF OF THE HANDOVER, AS AN EVENT.
    ///
    /// The nodal drive runs on THIS backend's stream and reads the jnet the
    /// segment's updjnet wrote on the SEGMENT's stream.  Exactness invariant 4
    /// requires that handover to be ordered by a synchronise or an event, and
    /// until now it was the synchronise -- the per-outer `sync_pre_nodal` drain,
    /// which is precisely what a host-free outer removes.  So the runner records
    /// an event on its stream after updjnet and hands it here; this makes the
    /// backend's stream wait on it, which is the same ordering with no host in
    /// it.
    ///
    /// ISSUED PER DRIVE AND OUTSIDE THE CAPTURE, for the reigv upload's reason:
    /// the event handle changes meaning every outer, and a wait recorded INTO
    /// the captured graph would freeze one outer's dependency into every replay.
    ///
    /// @param event an opaque cudaEvent_t already recorded on the caller's
    ///        stream, or nullptr for `no wait` -- the per-outer arm, every host
    ///        outer, and every path that still drains for itself.
    /// Returns false only when the wait itself failed.
    bool waitOnSegmentEvent(void* event);

    /// Receipt: how many routine per-outer transfers the sharing removed.
    [[nodiscard]] unsigned long long canonicalUploadsElided() const;
    [[nodiscard]] unsigned long long canonicalDownloadsElided() const;

    /// G0 receipt for the nodal kernel (RASBERY_GPU_NODAL): drive() calls
    /// completed on the device.
    static unsigned long long nodalDrivesSolved();

    /// Page-lock a host buffer this backend will repeatedly memcpy (same
    /// contract as CudaBatchArena::pinHost: idempotent, and leased rather than
    /// permanent -- the buffer's OWNER releases it with rasberyUnpinHost() in
    /// its destructor, see HostPinRegistry.h).  Returns true when the range is
    /// page-locked; false means the copies run pageable, which is legal and
    /// only slower.
    ///
    /// @param tag static string naming the CALL SITE, kept by the registry so a
    ///        RASBERY_PIN_DEBUG=1 refusal names both halves of the collision.
    ///        Defaulted so call sites that do not care still compile.
    static bool pinHost(const void* p, size_t bytes, const char* tag = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/// RASBERY_GPU_XSRECON, read once per process.  Stub builds return false.
bool rasberyGpuXsReconEnabled();

/// RASBERY_GPU_FLATXS, read once per process.  Stub builds return false.
bool rasberyGpuFlatXsEnabled();

/// RASBERY_GPU_MICX_RESIDENT, read once per process: WP15's deferred
/// micx/lmpx download.  Default OFF until the 238 runbook in
/// docs/WP15_MICX_RESIDENCY_20260830_KO.md has been run.  It is a B0 arm and it
/// is NOT in trajectory::kArmEnv -- it changes when a copy happens, never which
/// bytes it moves.  Stub builds return false.
bool rasberyGpuMicxResidentEnabled();

/// RASBERY_GPU_FLATXS_CTA, read once per process: WP5 stage B's CTA-per-node
/// flat-XS kernel.  DEFAULT ON since the v5 freeze (RASBERY_GPU_FLATXS_CTA=0 is
/// the off switch); the arm it changes is the RASBERY_GPU_FLATXS one, and it
/// changes it BIT FOR BIT -- the argument is written at the top of
/// src/FlatXsCtaKernel.cuh, gated by test/flatxs_device_replay.cu --cta, and
/// measured on both hosts (238 pricing block 12, 181 gates block 5).
/// Stub builds return false: a CUDA-less build has no flat-XS device arm at
/// all, so "the CTA arm is on" would be a claim about a kernel that is not
/// there.  The default is about which DEVICE kernel runs, not about whether a
/// device runs.
bool rasberyGpuFlatXsCtaEnabled();

/// Threads per CTA for that arm (RASBERY_GPU_FLATXS_CTA_THREADS; 64/128/256,
/// anything else clamps to flatxs::CTA_THREADS_DEFAULT).  PERFORMANCE ONLY:
/// every value on the ladder produces the same bytes, because each workspace
/// element's accumulation chain lives inside one lane whatever the block size.
int rasberyGpuFlatXsCtaThreads();

/// Receipt accessor mirroring XsReconBackend::nodesSolved for main.cpp.
unsigned long long rasberyGpuXsReconNodes();

/// Receipt accessor mirroring XsReconBackend::flatXsNodesSolved for main.cpp.
unsigned long long rasberyGpuFlatXsNodes();

/// RASBERY_GPU_XE, read once per process: the Rev.7.1 Task 13 split Xe arm
/// (evaluate / Anderson algebra / commit on the device).  Stub builds return
/// false.  DEFAULT OFF until the Gate A/B receipts adopt it.
bool rasberyGpuXeEnabled();

/// RASBERY_GPU_XE_TXN, read once per process: WP7 stage C's single-transaction
/// Xe step.  DEFAULT ON since the v5 freeze (RASBERY_GPU_XE_TXN=0 is the off
/// switch, and is the arm RASBERY_XE_FORMS_AUDIT=1 has to run on); the arm it
/// changes is the RASBERY_GPU_XE one, and it changes it bit for bit on both
/// hosts once the algebra channel is composed rather than mined (d25efe6).
/// Stub builds return false, for the same reason the CTA gate does.
bool rasberyGpuXeTxnEnabled();

/// The fixed partition count the device inner product is cut into
/// (RASBERY_GPU_XE_DOT_PARTITIONS, default xe::XE_DOT_PARTITIONS_DEFAULT).
/// One reproduces the host's serial fold exactly and is bit-gateable.
int rasberyGpuXeDotPartitions();

/// Receipt accessors mirroring XsReconBackend::xeEvaluations/xeCommits.
unsigned long long rasberyGpuXeEvaluations();
unsigned long long rasberyGpuXeCommits();

/// RASBERY_GPU_NODAL, read once per process.  Stub builds return false.
bool rasberyGpuNodalEnabled();

/// RASBERY_GPU_NODAL_FULL: run calculateEven on the device too, so the drive
/// is one uninterrupted device pipeline instead of the hybrid round-trip.
///
/// Deliberately an inline header function rather than another exported symbol:
/// Nodal.cpp and the backend BOTH have to agree on this flag (they used to
/// disagree -- Nodal.cpp tested presence, the backend tested truthiness, so
/// RASBERY_GPU_NODAL_FULL=0 put the two halves of the drive in different
/// modes and dropped the hybrid tail), and an inline definition keeps the
/// no-CUDA stub TU compiling untouched.  One static local, one process-wide
/// answer, same truthiness rule as the other flags.
inline bool rasberyGpuNodalFullEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("RASBERY_GPU_NODAL_FULL");
        if (v == nullptr) return false;
        const std::string s(v);
        return !(s.empty() || s == "0" || s == "off" || s == "OFF" ||
                 s == "false" || s == "FALSE");
    }();
    return on;
}

/// Arena width for the multi-instance NODAL batch (--batch-mode M), published
/// by main() next to rasberySetBatchWidth().
///
/// Deliberately NOT a call into rasberyBatchWidth(): CudaXsReconBackend.cu is
/// linked on its own into the device-consistency test target, without the CMFD
/// translation unit that defines that symbol, and the nodal arm has no other
/// reason to depend on the CMFD backend.  An inline function-local static gives
/// one process-wide value with no new exported symbol, so the CUDA build and
/// the no-CUDA stub build both keep linking unchanged -- the same reasoning as
/// rasberyGpuNodalFullEnabled above.
inline int& rasberyNodalBatchWidthRef() {
    static int width = 0;
    return width;
}
inline void rasberyNodalSetBatchWidth(int slots) {
    rasberyNodalBatchWidthRef() = slots > 0 ? slots : 0;
}
inline int rasberyNodalBatchWidth() { return rasberyNodalBatchWidthRef(); }

/// Receipt accessor mirroring XsReconBackend::nodalDrivesSolved for main.cpp.
unsigned long long rasberyGpuNodalDrives();

/// Rev.7.1 Task 18-lite receipt: the per-drive transfers the canonical binding
/// removed, in bytes, summed over the process.
///
/// FOUR PER DRIVE when the device outer segment holds the binding -- jnet and
/// flux up, jnet and phis back -- and they are invisible in the segment's own
/// receipt because the segment does not issue them; the nodal backend does.
/// Reporting them from here rather than inferring them from the outer count is
/// what makes the claim a measurement.
unsigned long long rasberyGpuNodalCanonicalElidedUploadBytes();
unsigned long long rasberyGpuNodalCanonicalElidedDownloadBytes();

// The process-wide host page-locking gate (rasberyHostPinningRef /
// rasberySetHostPinningEnabled / rasberyHostPinningEnabled) now lives in
// HostPinRegistry.h, included above, next to the lease lifecycle it gates:
// registration is no longer permanent, so the gate and the ownership registry
// are one contract and the owners that release leases (Geometry, Nodal, CMFD,
// BICGCMFD, XSSet) must be able to see it without pulling in a backend header.

} // namespace rasbery
