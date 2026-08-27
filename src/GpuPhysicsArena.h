#pragma once

// GpuPhysicsArena -- the one device allocation, Rev.7.1 plan Task 2 / Sec 4.
//
// THE HOLE THIS CLOSES.  Every existing device path in this tree allocates its
// own blocks and hands out raw pointers whose lifetime is the caller's problem.
// That is survivable while a launch is a launch.  It stops being survivable the
// moment a CUDA graph bakes those pointers into kernel arguments: a block that
// moves invalidates a captured graph, and the whole point of the graph is not
// to pay capture again.  So the arena takes ONE allocation, lays every byte out
// before any graph work happens, and never allocates again -- not on a refill,
// not on a phase transition, not on a resize, because there is no resize.
//
// WHAT IT DOES NOT DO.  It does not interpret anything.  importSlotAsync and
// exportSnapshotAsync are BYTE TRANSPORT: a region id, a host pointer, a byte
// count, a stream.  Every physical meaning lives in the layout calculator
// (GpuPhysicsArenaLayout.h, pure host, unit-tested with no CUDA) and in the
// phase kernels.  An arena that knew what a `phif` was would need a second
// opinion about it, and the two would drift.
//
// ALLOCATION POLICY (Sec 4.1).  A CUDA memory pool with its release threshold
// pinned at the block size, and one cudaMallocFromPoolAsync for the whole
// arena, synchronised before any pointer is published.  The pool is not there
// for suballocation -- there is none -- it is there so the block is
// stream-ordered and so the driver does not hand the pages back between
// statepoints.
//
// ADMISSION (Sec 4.4) FAILS LOUD.  10% of total VRAM held back for the driver,
// 10% of the request added for fragmentation, and a per-slot ceiling of 225 MiB
// from Sec 3.6.  A refusal is a refusal: the arena does NOT quietly reduce the
// slot count, because a 64-slot experiment that silently became a 24-slot
// experiment answers a different question than the one that was asked.
//
// SCRATCH IS PER SLOT AND PER PHASE (Sec 4.2).  scratch() takes the owning
// phase and, in a debug build, traps when that is not the slot's current phase.
// Release builds write the owner tag and do not read it, so the hot path pays
// nothing.  Rev.7's global-phase aliasing is gone: under an asynchronous
// scheduler slot A is in Outer while slot B is in Depletion, and a global
// lifetime would hand B the bytes A is still reading.

#include "GpuPhysicsArenaLayout.h"
#include "GpuPhysicsTypes.h"

#include <cstddef>
#include <iosfwd>
#include <string>

namespace rasbery::gpu {

/// Backend-neutral stream handle.  `cudaStream_t` is a pointer type, so it
/// round-trips through void* without a cast that loses anything; a HIP or SYCL
/// arm would do the same with its own queue handle.  Passing nullptr means the
/// default stream.
using GpuStreamHandle = void*;

class GpuPhysicsArena {
public:
    GpuPhysicsArena();
    ~GpuPhysicsArena();

    GpuPhysicsArena(const GpuPhysicsArena&)            = delete;
    GpuPhysicsArena& operator=(const GpuPhysicsArena&) = delete;

    /// Compute the layout, run Sec 4.4 admission, and take the ONE allocation.
    /// Returns false and leaves the arena unavailable on any refusal or CUDA
    /// error, with the reason in status().  Calling it twice is an error.
    ///
    /// After it returns true every pointer this object hands out is fixed for
    /// the lifetime of the arena -- which is the precondition for capturing a
    /// graph over them.
    bool reserve(const ArenaDims& dims);

    /// Release the block.  Safe to call on an unavailable arena.
    void release();

    [[nodiscard]] bool               available() const;
    [[nodiscard]] const std::string& status() const;
    [[nodiscard]] const ArenaOffsets& offsets() const;
    [[nodiscard]] const ArenaDims&    dims() const;
    [[nodiscard]] const ArenaAdmission& admission() const;

    /// Device base of the single block; nullptr when unavailable.
    [[nodiscard]] void* base() const;

    // --- addresses (all fixed after reserve()) -----------------------------

    [[nodiscard]] void* geometryRegion(GeometryRegion region) const;
    [[nodiscard]] void* libraryRegion(LibraryRegion region) const;
    [[nodiscard]] void* slotRegion(int slot, SlotRegion region) const;

    /// Sec 4.2: the request names the phase that owns the scratch.  In a debug
    /// build this traps when `phase` cannot own `id` at all, or when `phase` is
    /// not the slot's current phase.  In a release build it is the offset add
    /// and nothing else.
    [[nodiscard]] void* scratch(int slot, DevicePhase phase, ScratchId id) const;

    /// The half of the trap that needs no device read, exposed so the layout
    /// unit test can exercise it without CUDA.
    [[nodiscard]] static bool scratchPhaseAllowed(ScratchId id, DevicePhase phase) {
        return arenaScratchPhaseAllowed(id, phase);
    }

    // --- views -------------------------------------------------------------

    [[nodiscard]] DeviceGeometryView  geometryView() const;
    [[nodiscard]] DeviceXsLibraryView libraryView() const;
    /// A pure index rebase of the slot-0 view, exactly like nodalSlotView().
    [[nodiscard]] DeviceSlotView slotView(int slot) const;

    // --- byte transport ----------------------------------------------------

    bool importGeometryAsync(GeometryRegion region, const void* host, std::size_t bytes,
                             GpuStreamHandle stream);
    bool importLibraryAsync(LibraryRegion region, const void* host, std::size_t bytes,
                            GpuStreamHandle stream);
    bool importSlotAsync(int slot, SlotRegion region, const void* host, std::size_t bytes,
                         GpuStreamHandle stream);
    bool exportSnapshotAsync(int slot, SlotRegion region, void* host, std::size_t bytes,
                             GpuStreamHandle stream);

    /// Zero one slot's whole stride: the BULK arrays and the scratch bands.
    ///
    /// It cannot reach the four control structs, and that is structural rather
    /// than careful: they live in the contiguous control block below slot_base,
    /// outside every slot's stride.  While they were inside it, this memset ran
    /// straight over the defaults a refill had just written -- the schedule
    /// parameters, the reset epoch, the whole tenant identity -- so a refilled
    /// slot came back zeroed instead of initialised.  The layout gate asserts
    /// the control regions are disjoint from every slot stride.
    bool clearSlotAsync(int slot, GpuStreamHandle stream);

    // --- receipts ----------------------------------------------------------

    /// The Sec 9.3 memory receipt payload.
    [[nodiscard]] std::string receiptJson() const;
    /// Writes one `[RASBERY][GPU_ARENA] {...}` line.
    void emitReceipt(std::ostream& os) const;

private:
    struct Impl;
    Impl* _impl;
};

} // namespace rasbery::gpu
