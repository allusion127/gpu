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

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>

namespace rasbery {

namespace xsrecon {
struct BatchView;
}

namespace flatxs {
struct FlatXsView;
}

namespace nodal {
struct NodalView;
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

    /// G0 receipt for the nodal kernel (RASBERY_GPU_NODAL): drive() calls
    /// completed on the device.
    static unsigned long long nodalDrivesSolved();

    /// Page-lock a host buffer this backend will repeatedly memcpy (same
    /// contract as CudaBatchArena::pinHost: idempotent, never unregistered).
    static void pinHost(const void* p, size_t bytes);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/// RASBERY_GPU_XSRECON, read once per process.  Stub builds return false.
bool rasberyGpuXsReconEnabled();

/// RASBERY_GPU_FLATXS, read once per process.  Stub builds return false.
bool rasberyGpuFlatXsEnabled();

/// Receipt accessor mirroring XsReconBackend::nodesSolved for main.cpp.
unsigned long long rasberyGpuXsReconNodes();

/// Receipt accessor mirroring XsReconBackend::flatXsNodesSolved for main.cpp.
unsigned long long rasberyGpuFlatXsNodes();

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

/// Process-wide gate for host page-locking (cudaHostRegister), honoured by BOTH
/// XsReconBackend::pinHost and CudaBatchArena::pinHost.
///
/// Both of those register CALLER-owned host memory and never unregister it --
/// the registration is documented as permanent.  That contract is only sound
/// while every buffer they are handed also outlives the process, which holds
/// exactly when no Driver is ever destroyed before the run ends: a single deck
/// per process, or --batch-mode with one host worker per deck.
///
/// The moment a worker thread recycles (host_threads < jobs), Driver::Drive()
/// returns and destroys its stack-local Geometry/XSSet/BICGCMFD, freeing ranges
/// the CUDA driver still holds page-locked.  The allocator then hands those same
/// addresses to the next deck -- same thread arena, same sizes, same order -- so
/// the new deck's buffers alias a dead tenant's registration: cudaHostRegister
/// answers "already mapped", the DMA lands on the previous tenant's physical
/// pages, and any block that malloc served with mmap (and freed with munmap)
/// turns every later cudaMemcpyAsync on it into "invalid argument".
///
/// main() clears this gate for exactly those configurations.  Page-locking is a
/// throughput optimisation and nothing else: with it off the identical copies
/// run from pageable memory -- same bytes, same order, merely slower -- so the
/// numerical result is untouched.  Defaults to enabled, so every target that
/// does not call the setter (the replay/consistency executables, the stub build)
/// behaves exactly as before.
///
/// Deliberately an inline function-local static, for the same reason as
/// rasberyNodalBatchWidthRef() above: several test targets link
/// CudaXsReconBackend.cu without CudaBICGBackend.cu (and vice versa), and a
/// header-only flag adds no exported symbol to either.
inline bool& rasberyHostPinningRef() {
    static bool enabled = true;
    return enabled;
}
inline void rasberySetHostPinningEnabled(bool enabled) { rasberyHostPinningRef() = enabled; }
inline bool rasberyHostPinningEnabled() { return rasberyHostPinningRef(); }

} // namespace rasbery
