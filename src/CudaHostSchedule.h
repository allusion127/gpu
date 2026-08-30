#pragma once

// WP16 host-spin.  How this process WAITS on the device, decided once, before
// the CUDA context exists.
//
// THE OBSERVATION (238, 2026-08-30).  A batch of 8 processes x
// RASBERY_OMP_THREADS=8 on 24 CPUs shows ~79 % host CPU while ~92 % of process
// time is inside cudaStreamSynchronize (the XFER ledger says so).  Those two
// numbers cannot both be work.  CUDA's default device schedule is
// cudaDeviceScheduleAuto, and Auto SPINS when the number of active contexts is
// smaller than the number of logical cores -- which is exactly the 8-on-24
// shape -- so a thread that is "busy" in cudaStreamSynchronize is burning a
// core to poll a fence, and 8 processes' worth of pollers contend for 24 cores
// against the OpenMP teams that have real work.
//
// WHAT THIS KNOB IS.  `RASBERY_CUDA_SYNC_MODE = auto | spin | yield | blocking`
// maps one for one onto cudaSetDeviceFlags(cudaDeviceScheduleAuto / Spin /
// Yield / BlockingSync).  `blocking` parks the waiting thread on an interrupt
// instead of polling: the core is handed back, at the cost of a wake-up
// latency on every synchronise.  That trade is the measurement (see
// docs/WP16_HOST_SPIN_20260830_KO.md); this file only makes it takeable.
//
// B0 BY CONSTRUCTION.  Nothing here touches an operand, a kernel, a launch
// order or a stream.  It changes how the CALLING THREAD waits for a fence it
// was already waiting for, so every result is bit-identical in every mode.
//
// WHY IT LIVES IN ITS OWN TRANSLATION UNIT, CALLED FROM main().
// cudaSetDeviceFlags is only honoured while the process has NO CUDA context:
// once the runtime has created the primary context it returns
// cudaErrorSetOnActiveProcess and the flags in force are whatever the context
// was born with.  Nothing in this tree calls cudaSetDevice explicitly -- the
// context is created lazily by whichever backend's first runtime call wins
// (GpuPhysicsArenaCuda.cu:101, CudaXsReconBackend.cu:2817,
// CudaBICGBackend.cu:3114, CudaPprBackend.cu:1709, CudaCramBackend.cu:833) --
// so the only place that is provably earlier than all of them is the top of
// main().  main.cpp is not compiled by nvcc, hence the header/.cu/stub triple
// that the rest of the CUDA arms already use.

#include <string>

namespace rasbery {

/// What ApplyCudaHostSchedule() actually did, so main() can print one receipt.
///
/// The default values are the DEFAULT PATH's receipt: RASBERY_CUDA_SYNC_MODE
/// unset means no CUDA entry point is called at all, which is the only way to
/// promise "unchanged behaviour" rather than to assert it.
struct CudaHostScheduleReceipt {
    std::string mode      = "auto";      ///< resolved mode: auto|spin|yield|blocking
    std::string source    = "default";   ///< default | env | invalid
    std::string requested = "none";      ///< the cudaDeviceSchedule* symbol asked for
    std::string applied   = "unqueried"; ///< schedule bits cudaGetDeviceFlags reports back
    std::string rc        = "skipped";   ///< cudaSetDeviceFlags return, by name
    std::string note;                    ///< only when something needs saying
};

/// Set this process's device schedule from RASBERY_CUDA_SYNC_MODE.
///
/// Call ONCE, from main(), before anything else -- see the note above.  Unset
/// or unrecognised is a no-op that calls nothing.  A failure (a context that
/// already exists, a machine with no device) is reported in the receipt and is
/// never fatal: the run proceeds exactly as it would have without the knob.
CudaHostScheduleReceipt ApplyCudaHostSchedule();

} // namespace rasbery
