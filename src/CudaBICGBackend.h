#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace rasbery {

class Geometry;

enum SolveFlag : std::uint32_t {
    FLUX_CONVERGED     = 1u << 0,
    KEFF_CONVERGED     = 1u << 1,
    DHAT_CONVERGED     = 1u << 2,
    SEARCH_CONVERGED   = 1u << 3,
    NONFINITE_DETECTED = 1u << 4,
    NEGATIVE_FLUX      = 1u << 5,
    BICGSTAB_BREAKDOWN = 1u << 6,
    STALE_GENERATION   = 1u << 7,
    MAX_ITER_REACHED   = 1u << 8
};

/**
 * One cache-line status packet copied from device to host at a solver
 * observation boundary. Fields not owned by the linear solver are NaN/zero.
 */
struct alignas(64) DeviceSolveStatus {
    double keff;
    double flux_l2;
    double dhat_defect_max;
    double dhat_update_max;
    double search_residual;

    std::uint32_t flags;
    /// Reused as transport for the device-side tally of captured iterations
    /// that executed past the halt and were therefore no-ops.  The linear
    /// solver has no outer-iteration count of its own to report here.
    std::uint32_t outer_iter;
    std::uint32_t linear_iter;
    std::uint32_t material_gen;
    std::uint32_t operator_gen;
    std::uint32_t flux_gen;
};
static_assert(sizeof(DeviceSolveStatus) == 64);
static_assert(alignof(DeviceSolveStatus) == 64);

/**
 * Runtime evidence for strict GPU residency. The fields cover the common
 * backend contract even though this class currently owns only CMFD.
 */
struct BackendCounters {
    std::uint64_t xs_gpu_calls                         = 0;
    std::uint64_t xs_cpu_fallbacks                     = 0;
    std::uint64_t cmfd_gpu_calls                       = 0;
    std::uint64_t cmfd_cpu_fallbacks                   = 0;
    /// Slot-level CMFD operators assembled directly in the batch arena.
    std::uint64_t cmfd_assembly_gpu_calls              = 0;
    /// Sweep slots that retained the host-assembled operator path.
    std::uint64_t cmfd_assembly_cpu_fallbacks          = 0;
    /// Host diag/cc transfers avoided because assembly wrote resident arrays.
    std::uint64_t cmfd_diag_h2d_elided_bytes           = 0;
    std::uint64_t cmfd_cc_h2d_elided_bytes             = 0;
    /// psi round trip removed: the H2D bytes not re-pushed on a continuation
    /// launch, and the D2H bytes not pulled back on a non-exceptional one.
    std::uint64_t cmfd_psi_h2d_elided_bytes            = 0;
    std::uint64_t cmfd_psi_d2h_elided_bytes            = 0;
    /// Rev.7.1 Task 9 link 2: H2D bytes not copied because the device outer
    /// segment had already written the arena buffer the sweep reads.
    std::uint64_t cmfd_dhat_h2d_elided_bytes           = 0;
    std::uint64_t cmfd_resident_psi_h2d_elided_bytes   = 0;
    /// Cost of the CMFD flux mirror, in nanoseconds of LAUNCHER time: the
    /// adoptFluxMirror() shadow copy and the memcmp the next upload pays to
    /// decide whether it can skip.  Compare against the bytes that skip
    /// actually saved (cmfd_phi_h2d_elided_bytes) before believing in it.
    std::uint64_t cmfd_phi_mirror_ns                   = 0;
    std::uint64_t cmfd_phi_mirror_calls                = 0;
    std::uint64_t cmfd_phi_h2d_elided_bytes            = 0;
    /// Mirror maintenance skipped because the arena is single-slot (see
    /// phiMirrorEnabled()).
    std::uint64_t cmfd_phi_mirror_bypassed             = 0;
    std::uint64_t bicg_early_convergence_exits          = 0;
    std::uint64_t bicg_restarts                        = 0;
    std::uint64_t nodal_gpu_calls                      = 0;
    std::uint64_t nodal_cpu_fallbacks                  = 0;
    std::uint64_t th_gpu_calls                         = 0;
    std::uint64_t depletion_gpu_calls                  = 0;
    std::uint64_t bulk_h2d_calls_during_iteration      = 0;
    /// Bulk H2D uploads elided because the host buffer was bit-identical to
    /// what the device already holds (see CudaBICGBackend::reset).
    std::uint64_t bulk_h2d_skipped_during_iteration    = 0;
    /// Bytes actually pushed across PCIe by the bulk uploads above.
    std::uint64_t bulk_h2d_bytes_during_iteration      = 0;
    std::uint64_t bulk_d2h_calls_during_iteration      = 0;
    /// Bytes actually pulled back across PCIe by the bulk downloads above.  The
    /// D2H half of the per-statepoint transfer budget (plan Rev.4 Sec 8); its
    /// H2D counterpart is bulk_h2d_bytes_during_iteration.
    std::uint64_t bulk_d2h_bytes_during_iteration      = 0;
    std::uint64_t status_d2h_calls_during_iteration    = 0;
    std::uint64_t stream_sync_calls_during_iteration   = 0;
    std::uint64_t graph_launches                       = 0;
    std::uint64_t graph_reinstantiations               = 0;
    std::uint64_t graph_fallbacks                      = 0;
    /// Inner BiCGSTAB iterations captured into ONE graph launch -- the K of
    /// the iteration-batching knob (RASBERY_GPU_ITER_BATCH).  0 until the
    /// first launch; otherwise `max(1 + nmax, K)`.
    std::uint64_t iter_batch                           = 0;
    /// Graph launches that carried more than one captured iteration, i.e. the
    /// launches over which the per-iteration launch+sync cost was amortised.
    std::uint64_t batched_graph_launches               = 0;
    /// Captured iterations that ran after the device halt was raised and were
    /// therefore no-ops.  This is the direct evidence that the halt gating is
    /// complete: it is nonzero on every run where the inner loop converges
    /// early, and the accepted solution is unchanged by those iterations.
    std::uint64_t overrun_iterations                   = 0;
    /// 1 when the run DECLARED the mixed-precision inner BiCGSTAB
    /// (RASBERY_GPU_CMFD_FP32), 0 for the historical all-FP64 path.  It records
    /// the configuration, not the outcome: read it together with the next field.
    std::uint64_t fp32_active                          = 0;
    /// FP32 inner solves whose non-finite guard fired.  Each one was discarded
    /// (the flux kept its last finite iterate) and the arena reverted to the
    /// FP64 kernels for the rest of the process, so a nonzero value means the
    /// run FINISHED in fp64 whatever fp32_active says.
    std::uint64_t fp32_fallbacks                       = 0;
    /// WP20.2.  The captured refinement round CAP (rasbery::fp32::refineRounds()
    /// as the CMFD arm resolved it).  1 is the WP20 topology -- one FP64
    /// residual, one FP32 inner solve, one FP64 correction -- and it is what
    /// every arm without the refinement loop reports.
    std::uint64_t fp32_refine_cap                      = 0;
    /// Refinement rounds ACTUALLY entered, summed over slot-solves.  Divided by
    /// fp32_refine_solves this is the mean round count the receipt quotes; a run
    /// whose mean sits at the cap has a cap that is too small, and one whose
    /// mean sits at 1 is paying two scalar nodes per outer for nothing.
    std::uint64_t fp32_refine_rounds                   = 0;
    /// Slot-solves that reported a round count, i.e. the denominator of that
    /// mean.  Zero on every arm without a refinement loop, which is why the
    /// receipt reports the two sums and not a pre-divided figure: 0/0 is a
    /// missing measurement, not a mean of one.
    std::uint64_t fp32_refine_solves                   = 0;
};

/**
 * CUDA-resident two-group block-Jacobi BiCGSTAB backend.
 *
 * The CMFD matrix, Krylov vectors, and flux remain on the GPU for every
 * inner iteration. The converged/current flux is copied back explicitly at
 * the CMFD/nodal observation boundary.
 *
 * Since the multi-instance batch mode landed this is a *thin wrapper over a
 * one-slot batch arena*: there is exactly one kernel implementation, launched
 * with gridDim.y = 1 here and gridDim.y = M there.  Keeping a single body is
 * what makes "M = 1 batch mode is bit-identical to a plain single run" true by
 * construction rather than by manual synchronisation of two copies.
 */
class CudaBICGBackend {
private:
    class Impl;
    std::unique_ptr<Impl> _impl;

public:
    explicit CudaBICGBackend(Geometry& geometry);
    ~CudaBICGBackend();

    CudaBICGBackend(const CudaBICGBackend&)            = delete;
    CudaBICGBackend& operator=(const CudaBICGBackend&) = delete;

    [[nodiscard]] bool               available() const;
    [[nodiscard]] const std::string& status() const;
    [[nodiscard]] BackendCounters    counters() const;
    [[nodiscard]] DeviceSolveStatus  lastSolveStatus() const;

    /// Stage the operator, flux and source for one CMFD outer.  Uploads only:
    /// no kernel runs and no synchronisation happens here.
    void reset(const double* diag, const double* cc, const double* phi, const double* src);

    /// Enqueue the entire inner BiCGSTAB loop -- initial residual plus one
    /// unconditional iteration and `nmax` conditional ones -- as a single
    /// replayed CUDA graph.  The relative exit test `||r||/||r0|| < eps` runs
    /// on the device, so the host neither observes nor steers the loop.
    ///
    /// One call is therefore one graph launch and one status D2H carrying
    /// `1 + nmax` batched iterations; RASBERY_GPU_ITER_BATCH can capture more
    /// than that (inertly, see the counters), never fewer.
    void solveInner(int nmax, double eps);

    /// Drain the stream once per outer: fetch the flux and the status packet
    /// the graph queued, and raise the accumulated fatal condition if any.
    void synchronize(double* phi);
};

// ---------------------------------------------------------------------------
// Multi-instance (multi-state) batch arena.
//
// M independent RASBERY instances share one geometry, so they share the CMFD
// sparsity pattern (`neighbors`, `colors`) and differ only in the coefficients
// (diag / cc / src) and the flux.  The arena holds M *slots* of those
// coefficients back to back and runs the whole inner BiCGSTAB loop for every
// slot in one grid: block (b, m) of the batched launch does exactly the work
// that block b of instance m's single-instance launch would have done.
//
//   *** The bit-identity rule: gridDim.x is per-instance, never re-chunked. ***
//
// The only order-sensitive operation in the solver is the dot product, whose
// partition is `chunk = ceil(n / gridDim.x)`.  Adding the batch axis as
// gridDim.y leaves gridDim.x -- and therefore the chunking, the operand
// pairing and the summation order -- exactly as it was.  Batching is a grid
// *re-layout*, not a re-association, so every slot reproduces the single-run
// answer bit for bit no matter which other slots ride along.
//
// A corollary the host side depends on: because the answer does not depend on
// the batch composition, the arena is free to launch with whichever instances
// happen to be waiting.  Instances never have to march in lock step.
// ---------------------------------------------------------------------------
class CudaBatchArena {
private:
    class Impl;
    std::unique_ptr<Impl> _impl;

public:
    CudaBatchArena(Geometry& geometry, int slots);
    ~CudaBatchArena();

    CudaBatchArena(const CudaBatchArena&)            = delete;
    CudaBatchArena& operator=(const CudaBatchArena&) = delete;

    [[nodiscard]] bool               available() const;
    [[nodiscard]] const std::string& status() const;
    [[nodiscard]] int                slots() const;
    [[nodiscard]] BackendCounters    counters() const;

    /// True when @p geometry has the same CMFD shape the arena was built for.
    /// The batch mode is restricted to a single geometry on purpose (the DB
    /// use case varies state, not the core).
    [[nodiscard]] bool compatible(Geometry& geometry) const;

    /// Claim/return one instance slot.  Thread-safe; -1 when the arena is full.
    int  acquireSlot();
    void releaseSlot(int slot);

    /// Record this outer's host buffers for @p slot.  No CUDA call and no lock:
    /// the owning thread only compares against its private upload shadow.
    void stage(int slot, const double* diag, const double* cc,
               const double* phi, const double* src);

    /// Inner-loop budget for @p slot.  `nmax` fixes the graph topology and must
    /// therefore agree across the batch; `eps` is per-slot device state.
    void setInner(int slot, int nmax, double eps);

    /// Rendezvous + launch + drain.  Returns once @p phi holds this slot's
    /// updated flux.  Whichever thread completes the waiting set launches for
    /// all of them; the rest sleep on a condition variable.
    void solve(int slot, double* phi);

    /// Host<->device contract of one device-resident CMFD sweep run
    /// (RASBERY_GPU_CMFD_SWEEP).  Inputs describe the state at delegation;
    /// outputs return the sweep loop's results.  `state`: 1 converged, 2 the
    /// wiel gamma degenerated and the host must finish the current sweep with
    /// the Rayleigh branch (gammad/gamman/err_acc carry the exported sums),
    /// 3 the sweep budget was spent, 0 only the launch unroll ran out.
    struct CmfdSweepIO {
        const double* chif  = nullptr; ///< [ig*nxyz+l]
        const double* xsnf  = nullptr; ///< [ig*nxyz+l]
        const double* xsrf  = nullptr; ///< [ig*nxyz+l], device assembly input
        const double* xssm  = nullptr; ///< [(igs*ng+ige)*nxyz+l]
        const double* dtil  = nullptr; ///< [surface*ng+ig]
        const double* dhat  = nullptr; ///< [surface*ng+ig]
        const double* vol   = nullptr; ///< [l]
        double*       udiag = nullptr; ///< [l*ng2+ige*ng+igs], host fallback
        double*       psi   = nullptr; ///< [l], in/out
        /// Has the HOST written `psi` since the previous launch of this drive?
        ///
        /// It used to be downloaded after every launch and re-uploaded before
        /// the next one, which is a 2 x nxyz-double round trip per launch to
        /// hand the device back exactly the bytes it just produced.  The host
        /// only touches psi at a drive boundary (CMFD::updpsi regenerates it
        /// from the flux before every BICGCMFD::drive, and nothing between two
        /// launches of one drive writes it), so the first launch of a drive
        /// uploads and the rest skip.  Skipping leaves the device's own psi in
        /// place, which is bit-for-bit what the round trip put back.
        bool          psi_dirty = true;
        /// Rev.7.1 Task 9 link 2: the device outer segment already wrote these
        /// INTO THE ARENA, so the H2D would be copying the host's stale twin over
        /// the authoritative bytes.
        ///
        /// THIS IS A POINTER HANDOFF AND NOT A CACHE.  The segment's upddhat
        /// kernel writes dhat_dev for this slot directly -- the same buffer
        /// cmfd_assemble_operator_2g reads -- so `skip the upload` here is not an
        /// optimisation that could be wrong, it is the only correct action: the
        /// host array is one outer behind by construction.
        ///
        /// dhat is the expensive one.  It is the ONE sweep input pushed
        /// unconditionally every outer (nsurf*ng doubles, ~416 KiB at APR1400
        /// size) because it changes after every nodal correction, so comparing a
        /// mirror would cost more than the copy.  When the segment owns it there
        /// is nothing to compare and nothing to copy.
        bool          dhat_device_resident = false;
        bool          psi_device_resident  = false;
        /// Build diag/cc/udiag in the arena before the resident sweep graph.
        bool          device_assembly = false;
        double eigv = 0, reigv = 0, reigvs = 0, errl2 = 0;
        double epsl2 = 0, eshift = 0;
        int    sweep_budget = 0, icmfd_budget = 0, icmfd_done = 0, ngxyz = 0;
        // outputs
        int    sweeps_done = 0, state = 0, negative_last = 0;
        double gammad = 0, gamman = 0, err_acc = 0;
    };

    /// Page-lock a host buffer the arena will repeatedly memcpy.  Pageable
    /// cudaMemcpyAsync stages through the driver and blocks the launcher;
    /// pinned transfers run at bus speed and genuinely overlap.  Idempotent
    /// (an already-registered range is reused, not re-registered), and LEASED:
    /// the buffer's owner releases it with rasberyUnpinHost() in its destructor
    /// so a recycled Driver worker cannot inherit a dead deck's registration.
    /// See HostPinRegistry.h.  Returns true when the range is page-locked.
    ///
    /// @param tag static string naming the CALL SITE, kept by the registry for
    ///        the RASBERY_PIN_DEBUG=1 collision report.
    bool pinHost(const void* p, size_t bytes, const char* tag = nullptr) const;

    /// Record one drive()'s sweep inputs for @p slot.  No CUDA call, no lock.
    /// Rev.7.1 Task 9 link 2: the DEVICE addresses of one slot's CMFD state.
    ///
    /// WHY THIS IS A HANDOFF AND NOT A COPY.  Every array here already has the
    /// layout the host uses, so the device outer segment can write the buffers
    /// the sweep reads with no transpose and no staging:
    ///
    ///   phi  [l*ng + ig]      -- node-major, byte-identical to Geometry::Phif;
    ///                            cmfd_wiel_terms reads f[l*2+0], f[l*2+1].
    ///   psi  [l]              -- cmfd_wiel_terms writes ps[l].
    ///   dtil/dhat [ls*ng+ig]  -- CMFD.h's own order.
    ///   xsnf [ig*nxyz + l]    -- group-major; x0 = base, x1 = base + nxyz.
    ///
    /// That coincidence is not luck: the arena was built to share the host's
    /// addressing so the Class B0 bodies could be scored against the CPU loops.
    /// It is what makes link 2 a pointer handoff rather than a second layout.
    ///
    /// The pointers are fixed for the arena's life, so a caller may bind them
    /// once.  `valid` is false when the arena is unavailable or the slot is out
    /// of range, in which case every pointer is null.
    struct CmfdResidentView {
        double*       phi  = nullptr;
        double*       psi  = nullptr;
        double*       dtil = nullptr;
        double*       dhat = nullptr;
        const double* xsnf = nullptr;
        int  nxyz  = 0;
        int  ngxyz = 0;
        int  nsurf = 0;
        bool valid = false;
    };
    [[nodiscard]] CmfdResidentView residentView(int slot) const;

    void stageSweeps(int slot, const CmfdSweepIO& io);

    /// Rendezvous + one multi-sweep graph launch + drain; same batching
    /// contract as solve() but in its own rendezvous domain (a batch never
    /// mixes plain solves with sweep runs -- the graphs differ).
    void solveSweeps(int slot, double* phi, CmfdSweepIO& io);

    // -----------------------------------------------------------------------
    // Rev.7.1 Task 10 part 2: the sweep as a STREAM-ORDERED ENQUEUE
    // -----------------------------------------------------------------------
    //
    // WHAT solveSweeps COSTS THAT A SEGMENT CANNOT AFFORD.  It takes the arena
    // mutex, joins a rendezvous, lingers for absent participants, launches,
    // drains the stream, runs the per-slot drain bookkeeping and finishes with
    // adoptFluxMirror -- an n-double host memcpy.  All of that is per OUTER, and
    // it is exactly the host rendezvous the device outer segment exists to
    // remove; while the sweep hook calls it, OuterSegmentHooks::sweep_synchronizes
    // has to stay true and the segment budget is forced to one.
    //
    // enqueueSweeps is the same launch with the rendezvous and the drain taken
    // out: the caller owns the stream (sweepStream(), so segment kernels and
    // sweep kernels are ordered against each other by construction) and observes
    // when IT is ready.  Single participant only -- a batch has a rendezvous
    // precisely because it has more than one arrival, and the segment refuses
    // batch mode anyway.

    /// The arena's stream, so a caller can order its own work against the sweep
    /// without an event.  Null when the arena is unavailable.
    [[nodiscard]] void* sweepStream() const;

    /// Where the device publishes what the outer segment has to rank, so the
    /// normal path needs no host readback at all.  Every pointer is DEVICE
    /// memory owned by the caller; a null one is simply not written.
    struct CmfdSweepProbeSink {
        double*        eigv     = nullptr; ///< the sweep's eigenvalue
        double*        residual = nullptr; ///< the sweep's L2 flux residual
        std::uint32_t* negative = nullptr; ///< negative-flux entries of the last sweep
        std::uint32_t* rayleigh = nullptr; ///< sweep state 2, the gamma hand-back
        /// CLEARED, never set, and that is the point.  The non-finite flag is
        /// raised by the segment's own input-refresh kernel, which is the one
        /// place that holds eigv and residual at once -- but it has to be raised
        /// PER OUTER.  The host probe publish it replaces wrote a zeroed struct
        /// every outer; a verdict that wrote only the four live fields would let
        /// one non-finite outer latch the flag for the rest of the process, so
        /// the next armed segment would fail its first outer on a signal from a
        /// solve that had already been abandoned.
        std::uint32_t* nonfinite = nullptr;
        /// The segment's per-slot halt word, indexed by `halt_slot`.
        ///
        /// READ as well as written: a halted outer must not run its sweep at
        /// all, and the re-seed kernel is the one place inside the sweep that
        /// can see the segment's halt before the graph starts.  Written only to
        /// latch an incomplete drive (sweep state 0 or 2), which ends the outer
        /// where it stands rather than letting the body run on a half sweep.
        std::uint32_t* halt      = nullptr;
        int            halt_slot = 0;

        // -------------------------------------------------------------------
        // Rev.7.1 Task 10 part 3: THE SWEEP, OBSERVED ONCE PER SEGMENT
        // -------------------------------------------------------------------
        //
        // WHAT THE PER-OUTER OBSERVATION COSTS.  BICGCMFD::finishDrive reads
        // this launch's scalar block out of pinned host memory the stream
        // filled, so the stream has to be DRAINED first -- `sync_pre_nodal` in
        // the receipt, one per outer on every arm and every budget, and the
        // whole of what a host-free outer was blocked on.
        //
        // WHAT THE HOST ACTUALLY DOES WITH WHAT IT READS, on the normal path
        // (sweep state 1 or 3): it copies eigv/residual into locals the segment
        // already reconstructs from DeviceSlotState at the exit, and it advances
        // three counters from `icmfd_done`.  Only the counters are irreplaceable,
        // and they are a SUM -- so the device can keep the sum.
        //
        // WHAT IS NOT REPLACEABLE is the exceptional launch (state 0, the slot
        // budget ran out mid-retry; state 2, the Wielandt gamma degenerated).
        // There the host has to finish the drive verbatim, from THAT launch's
        // scalar block -- which the launches enqueued behind it would have
        // overwritten.  So the accumulator keeps a copy of it.
        struct Accum {
            double attempts    = 0; ///< sum of kIcmfdDone over observed launches
            double sweeps      = 0; ///< sum of kSweepsDone
            double launches    = 0; ///< launches whose verdict actually ran
            double state       = 0; ///< the last observed launch's sweep state
            double negative    = 0; ///< the last observed launch's negative census
            double exceptional = 0; ///< 1 once some launch ended in state 0 or 2
            /// The abandoned launch's whole scalar block, saved where it is
            /// still readable.  Width is asserted against kSweepCount in the .cu.
            double saved[19]   = {};
        };
        /// Device pointer to one Accum, or null for the per-outer arm.
        ///
        /// ZEROED BY THE CALLER at the segment entry, and only there: it is a
        /// SEGMENT-scoped total, and a launch that finds the halt already up
        /// contributes nothing to it (the verdict kernel returns first), which
        /// is exactly the definition the host reconstruction needs.
        Accum*         accum = nullptr;
        /// Take this launch's eigv/reigv/reigvs/errl2 from the probe above
        /// rather than from the host's staged copy.
        ///
        /// FALSE ON THE FIRST OUTER OF A SEGMENT.  There the host's eigenvalue
        /// IS the current one -- it came out of the previous segment's exit
        /// observation -- while the probe may still describe a drive several
        /// host outers old.  From the second outer on, the probe holds the
        /// PREVIOUS outer's verdict, which is precisely what finishDrive would
        /// have handed the host.
        bool           patch_from_probe = false;
    };

    /// Enqueue one resident sweep run on sweepStream() and RETURN.
    ///
    /// Nothing is drained and nothing is read back: `io`'s outputs stay
    /// unfilled.  The caller synchronises when it is ready and then calls
    /// readSweepObservation() to learn whether the drive finished.
    ///
    /// Returns false without enqueueing anything when the arena cannot serve a
    /// single-participant launch, in which case the caller must take solveSweeps.
    ///
    /// Rev.7.1 Task 18: @p caller_stream is the stream the CALLER's own work is
    /// on.  Pass sweepStream() (or nullptr) when the caller has bound the arena
    /// stream itself -- the single-deck arrangement -- and the two are already
    /// ordered by construction.  Pass a DIFFERENT stream and the launch is
    /// joined to it by an event pair, which is what lets M device outer segments
    /// keep M private streams and still share this one arena: a capture
    /// swallows every enqueue on the stream it is open on, so M segments cannot
    /// share the arena stream, and without the join nothing would order a
    /// segment's updpsi against the sweep that reads its psi.
    bool enqueueSweeps(int slot, double* phi, const CmfdSweepIO& io,
                       const CmfdSweepProbeSink& probe, void* caller_stream = nullptr);

    /// Copy the sweep outputs of the last enqueueSweeps into @p io.
    ///
    /// ONLY VALID AFTER A SYNCHRONISE of sweepStream(): the D2H that fills the
    /// staging block rode that stream.  This is a host read of pinned memory,
    /// not a transfer, so it costs a load per field.
    void readSweepObservation(int slot, CmfdSweepIO& io) const;

    /// The post-synchronise half of an enqueueSweeps launch: the mirror commits,
    /// the per-slot status absorb and the flux mirror that drain() would have
    /// done, plus readSweepObservation.  THE CALLER MUST HAVE SYNCHRONISED
    /// sweepStream() -- this deliberately does not, because the whole point of
    /// the enqueue path is that the segment owns when that happens.
    ///
    /// Returns false when the device flagged this slot's flux non-finite.
    bool finishSweeps(int slot, CmfdSweepIO& io);

    /// Rev.7.1 Task 10 part 3: the same bookkeeping for a WHOLE SEGMENT of
    /// launches that were never observed one at a time.
    ///
    /// The two differences from finishSweeps, and both are forced by the same
    /// fact -- the staging block belongs to whichever launch was enqueued LAST,
    /// which on a segment that halted is one whose every kernel was masked:
    ///
    ///   * readSweepObservation is NOT called.  The caller reconstructs eigv and
    ///     the residual from DeviceSlotState and the counters from the
    ///     accumulator, which are the only two records that survive a masked
    ///     launch.
    ///   * @p state comes from the accumulator, not from the block, and it is
    ///     what decides whether the Rayleigh hand-back's operator downloads run.
    ///
    /// Returns false when the device flagged this slot's flux non-finite.
    bool finishSweepsDeferred(int slot, int state);

    /// Rev.7.1 Task 10 part 3: unpack the accumulator's saved scalar block into
    /// the OUTPUT half of an IO block.
    ///
    /// The one place the sweep block's slot map is spelled outside the backend
    /// would be BICGCMFD, which cannot see it -- the map is an anonymous-
    /// namespace enum in the .cu, and putting a second copy of it in a header
    /// is how two spellings of one layout start disagreeing.  So the caller
    /// hands the bytes back and this fills the fields readSweepObservation
    /// would have filled, from exactly the same indices.
    static void unpackSavedSweepBlock(const CmfdSweepProbeSink::Accum& acc,
                                      CmfdSweepIO& io);

    /// Drain sweepStream().  For the caller that has to fall back to a BLOCKING
    /// drive after issuing stream-ordered work of its own -- the Wielandt
    /// warm-up inside a device outer segment -- where the host loop is about to
    /// read arrays those copies are still filling.
    void syncSweepStream();

    /// One JSON line of how full the batches actually were.  The batch mode is
    /// only worth its complexity when the mean width is close to the slot
    /// count, so this is the number to read first when the speed-up disappoints.
    void reportBatchOccupancy(const char* tag) const;

private:
    /// Shared rendezvous body of solve()/solveSweeps(); kind selects the
    /// batch domain and the launch sequence.
    void solveCommon(int slot, double* phi, int kind);

public:
};

// ---------------------------------------------------------------------------
// Rev.7.1 Task 6 Step 3: the sweep graph is CAPACITY, not configuration.
// ---------------------------------------------------------------------------

/// What a captured device-resident CMFD sweep graph is good for.
///
/// THE BUG THIS REPLACES.  The graph was keyed on the exact `unroll` it was
/// captured at, and `unroll` is the REMAINING sweep budget (`_ncmfd - iout`,
/// BICGCMFD.cpp:394) with _ncmfd = 5 (Driver.h:2114).  So it walked 5,4,3,...
/// and back to 5 at the next drive(): the topology was destroyed and rebuilt
/// continuously, `graph_reinstantiations` climbed for the whole run, and Task 10's
/// instantiation gate could not pass.
///
/// THE FIX.  `unroll` became a device scalar (kSweepSlotBudget), so a graph with
/// MORE slots than a launch may spend is not merely acceptable -- it is exactly
/// equivalent, because the excess slots halt in cmfd_sweep_begin before reading
/// or writing anything.  The cache therefore only has to GROW, which it does at
/// most once per run: the first launch of every drive() asks for the largest
/// budget there is (iout == 0).
///
/// nmax IS STILL AN EXACT KEY, deliberately.  A deeper nmax capture would
/// over-iterate: `force_halt` is placed at capture time from `1 + nmax`
/// (CudaBICGBackend.cu, enqueue_outer), so a graph captured for a larger nmax
/// runs more inner iterations than a smaller request wants.  That is safe to
/// leave exact because nmax is a PROCESS CONSTANT -- it is read once from
/// RASBERY_BICG_NMAX in the BICGCMFD constructor, setIterLim has no callers, and
/// batch mode already refuses a non-uniform nmax -- so it contributes at most
/// one instantiation per run.  The contract test pins that reasoning.
struct SweepGraphCapacity {
    int nmax      = -1; ///< exact key: inner BiCGSTAB budget (a process constant)
    int slots     = -1; ///< capacity key: sweep slots CAPTURED
    int precision = -1; ///< exact key: the FP32 fallback changes the topology
    /// Exact key: grid.y, which a captured graph BAKES.  It is the arena width
    /// unless RASBERY_GPU_CMFD_COMPACT is on, in which case it is the bucket
    /// for the arrival width and each bucket needs its own instantiation.
    /// Unlike `slots` above this is NOT a capacity: a graph captured 32 lanes
    /// wide dispatches 32 lanes whatever the map says, so a narrower launch
    /// would pay for padding blocks the compaction exists to remove.
    int lanes     = -1;

    /// Can the captured graph serve a launch wanting `want_slots` slots?
    [[nodiscard]] constexpr bool serves(int want_nmax, int want_slots,
                                        int want_precision, int want_lanes) const {
        return slots >= 0 && nmax == want_nmax && precision == want_precision &&
               lanes == want_lanes && slots >= want_slots;
    }

    /// Depth to capture at.  Never shallower than what is already captured, so
    /// the capacity ratchets and settles instead of oscillating.
    [[nodiscard]] constexpr int captureDepth(int want_slots) const {
        return slots > want_slots ? slots : want_slots;
    }
};

// ---------------------------------------------------------------------------
// Process-wide batch-mode plumbing.  main() declares the batch width once,
// and the first BICGSolver that is constructed afterwards builds the arena
// from its own Geometry (the shape is not known before a deck is read).
// ---------------------------------------------------------------------------

/// Declare the batch width for this process.  0 disables the batch path.
void rasberySetBatchWidth(int slots);

/// The width declared above (0 when batch mode is off).
[[nodiscard]] int rasberyBatchWidth();

// ---------------------------------------------------------------------------
// RESOLVED (was: "LOCAL-TESTING CAVEAT: the resident sweep tests need sm_120")
// ---------------------------------------------------------------------------
//
// This block used to say the device-resident CMFD sweep path is not run-to-run
// deterministic on the sm_61 development card (GTX 1080 Ti, WSL2) -- up to
// 90 pcm of drift -- while RASBERY_GPU=1 alone and --batch-mode 1 without
// RASBERY_GPU_CMFD_SWEEP were exactly 0.000 pcm, and concluded that the drift
// was a property of that card rather than of this code, so the sweep path could
// only be judged on sm_120.
//
// THAT CONCLUSION WAS WRONG, and believing it cost a campaign week.  The drift
// was ours: issueSweepUploads built the participation mask in the page-locked
// `host_active`, uploaded it with cudaMemcpyAsync, and then INVERTED THAT SAME
// BUFFER IN PLACE to make the sweep_halt mask -- a host write to the source of
// an unsynchronised DMA.  Whether the copy engine had already read the bytes
// decided whether `device_active` arrived correct or all-zero, so one binary
// gave different answers on different runs; a zeroed `active` masks that
// sweep's whole BiCGSTAB inner loop while the Wielandt tail still advances psi
// and the eigenvalue, and the drive converges to a neighbouring iterate.  The
// symptom was architecture-CORRELATED only because the driver's decision to
// stage a small pinned copy inline or defer it depends on the queue state.
//
// It reproduces on sm_61 in four seconds (i-SMR CY01, resident-single arm: the
// non-finite abort, 3/3), and with the fix that arm is 5/5 bit-identical AND
// bit-identical to the RASBERY_GPU_CMFD_SWEEP=0 reference.  So: the sweep path
// IS testable locally, and an sm_61 A/B on it is meaningful again.  See
// tools/test_cmfd_async_h2d_snapshot_contract.py for the invariant that keeps
// it that way.

/// Rev.7.1 Task 6: RASBERY_GPU_CMFD_RESIDENT_SINGLE, default OFF.
///
/// With it set, a run WITHOUT --batch-mode still builds the arena -- at width 1
/// -- so a single instance reaches the resident device assembly and the
/// device-resident sweeps.  Before this, `_ls->arena()` was null outside batch
/// mode and canUseDeviceAssembly() (BICGCMFD.cpp:207-217) therefore refused,
/// which meant the whole resident path was reachable only by asking for a batch
/// of one.  Nothing about the kernels changes: the same BatchCore drives one
/// physical slot, so there is no second code path to keep in step.
[[nodiscard]] bool rasberyResidentSingleCmfd();

/// The process arena, created on first call from @p geometry.  Throws when a
/// later instance presents an incompatible geometry.
CudaBatchArena* rasberyBatchArena(Geometry& geometry);

/// Tear the arena down and report its aggregated counters (end of run).
void rasberyReleaseBatchArena();

} // namespace rasbery
