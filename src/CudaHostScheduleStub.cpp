// CPU-only builds: same symbol as CudaHostSchedule.cu, no CUDA anywhere, so
// main() never needs an #ifdef.  Mirrors CudaXsReconBackendStub.cpp.
//
// The receipt still reports what the operator ASKED for.  A CPU-only binary
// that silently printed `mode:"auto"` while RASBERY_CUDA_SYNC_MODE=blocking was
// exported would make a campaign row look like a device arm that never ran on
// a device; `rc:"no-cuda"` is the line that stops that.

#include "CudaHostSchedule.h"

#include <cstdlib>
#include <cstring>

namespace rasbery {

CudaHostScheduleReceipt ApplyCudaHostSchedule() {
    CudaHostScheduleReceipt receipt;
    receipt.applied = "n/a";
    receipt.rc      = "no-cuda";

    const char* env = std::getenv("RASBERY_CUDA_SYNC_MODE");
    if (env == nullptr || *env == '\0') return receipt;

    if (std::strcmp(env, "auto") == 0 || std::strcmp(env, "spin") == 0 ||
        std::strcmp(env, "yield") == 0 || std::strcmp(env, "blocking") == 0) {
        receipt.mode   = env;
        receipt.source = "env";
    } else {
        receipt.source = "invalid";
        receipt.note   = std::string("unrecognised RASBERY_CUDA_SYNC_MODE=\"") + env +
                       "\"; expected auto|spin|yield|blocking. Ran unchanged (auto).";
    }
    return receipt;
}

} // namespace rasbery
