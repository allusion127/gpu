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
#include <memory>
#include <string>

namespace rasbery {

namespace xsrecon {
struct BatchView;
}

namespace flatxs {
struct FlatXsView;
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
    bool solve(const xsrecon::BatchView& host, unsigned long long micx_generation,
               double* max_change_out);

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
                     unsigned long long ref_generation, bool mark_micx_resident);

    /// Total fuel nodes processed on the device by this process (all
    /// instances).  Zero means the device path never ran, whatever the flag
    /// said -- the G0 validity receipt.
    static unsigned long long nodesSolved();

    /// Same receipt for the flat-XS kernel (RASBERY_GPU_FLATXS).
    static unsigned long long flatXsNodesSolved();

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

} // namespace rasbery
