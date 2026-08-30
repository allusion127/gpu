// CUDA arm of CudaHostSchedule.h -- WP16 host-spin.
//
// The whole translation unit is ONE cudaSetDeviceFlags and ONE
// cudaGetDeviceFlags.  It deliberately contains no allocation, no stream, no
// device query and no kernel: any of those would create the primary context
// that cudaSetDeviceFlags has to precede, and this file's entire purpose is to
// be the thing that runs first.

#include "CudaHostSchedule.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

namespace rasbery {

namespace {

/// The four schedule bits, by the name the knob uses.
std::string scheduleName(unsigned int flags) {
    switch (flags & cudaDeviceScheduleMask) {
        case cudaDeviceScheduleAuto:         return "auto";
        case cudaDeviceScheduleSpin:         return "spin";
        case cudaDeviceScheduleYield:        return "yield";
        case cudaDeviceScheduleBlockingSync: return "blocking";
        default:                             return "unknown";
    }
}

} // namespace

CudaHostScheduleReceipt ApplyCudaHostSchedule() {
    CudaHostScheduleReceipt receipt;

    const char* env = std::getenv("RASBERY_CUDA_SYNC_MODE");
    if (env == nullptr || *env == '\0') {
        // THE DEFAULT PATH CALLS NOTHING.  cudaSetDeviceFlags(Auto) would be a
        // no-op in theory, and in practice it is still a CUDA entry point in a
        // process that may never have wanted one (a CPU-only deck, a --help, a
        // machine with a broken driver).  "Unchanged behaviour" is only
        // provable if the default does not enter the runtime at all.
        return receipt;
    }

    unsigned int flags = cudaDeviceScheduleAuto;
    if (std::strcmp(env, "auto") == 0) {
        receipt.mode      = "auto";
        receipt.requested = "cudaDeviceScheduleAuto";
        flags             = cudaDeviceScheduleAuto;
    } else if (std::strcmp(env, "spin") == 0) {
        receipt.mode      = "spin";
        receipt.requested = "cudaDeviceScheduleSpin";
        flags             = cudaDeviceScheduleSpin;
    } else if (std::strcmp(env, "yield") == 0) {
        receipt.mode      = "yield";
        receipt.requested = "cudaDeviceScheduleYield";
        flags             = cudaDeviceScheduleYield;
    } else if (std::strcmp(env, "blocking") == 0) {
        receipt.mode      = "blocking";
        receipt.requested = "cudaDeviceScheduleBlockingSync";
        flags             = cudaDeviceScheduleBlockingSync;
    } else {
        // A typo must not silently become an arm.  Report it and take the
        // default path -- the run is still valid, and the receipt says the
        // campaign row is an `auto` row, not the row the operator typed.
        receipt.source = "invalid";
        receipt.note   = std::string("unrecognised RASBERY_CUDA_SYNC_MODE=\"") + env +
                       "\"; expected auto|spin|yield|blocking. Ran unchanged (auto).";
        return receipt;
    }

    receipt.source = "env";

    // BOTH OUTCOMES ARE REPORTED, NEITHER IS FATAL.  cudaSetDeviceFlags before
    // any context succeeds and the primary context is later born with these
    // bits; after a context exists it returns cudaErrorSetOnActiveProcess and
    // the bits in force are the ones the context already has.  `applied` is
    // read back rather than assumed, so the receipt distinguishes "asked and
    // got it" from "asked too late" without the reader having to know which
    // call site created the context.
    const cudaError_t rc = cudaSetDeviceFlags(flags);
    receipt.rc           = cudaGetErrorName(rc);

    unsigned int      actual = 0;
    const cudaError_t query  = cudaGetDeviceFlags(&actual);
    receipt.applied          = (query == cudaSuccess) ? scheduleName(actual) : "unqueried";

    // A failure above also latches into the runtime's last-error slot, and the
    // next backend to call cudaGetLastError() would read OUR rc and blame its
    // own kernel for it.  Clear it here; the receipt already carries it.
    cudaGetLastError();
    return receipt;
}

} // namespace rasbery
