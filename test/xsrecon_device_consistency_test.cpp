// Device-arm consistency harness: the CUDA backend must reproduce the CPU
// reference loop bit-identically on the same synthetic data the host harness
// uses.  This is the gate the host harness cannot provide -- nvcc's floating
// point versus g++'s -- run elementwise on one call, where a full-deck A/B
// can only show that thousands of downstream bits drifted.
//
// CUDA builds only (linked against CudaXsReconBackend.cu).  Sets
// RASBERY_GPU_XSRECON=1 itself before the backend reads it.
//
// Usage: rasbery_xsrecon_device_consistency [nxyz]   (default 4096)

#include "CudaXsReconBackend.h"
#include "xsrecon_test_common.h"

#include <cstdlib>

int main(int argc, char** argv) {
    // Before anything caches the flag.
    setenv("RASBERY_GPU_XSRECON", "1", 1);

    const int nxyz = (argc > 1) ? std::atoi(argv[1]) : 4096;
    if (nxyz <= 0) {
        std::printf("bad nxyz\n");
        return 2;
    }

    int           total_diffs = 0;
    std::uint64_t seed        = 0x9E3779B97F4A7C15ULL;
    for (double relax : {1.0, 0.5}) {
        g_state      = seed;
        Arrays ref_a = makeArrays(nxyz);
        g_state      = seed;
        Arrays dev_a = makeArrays(nxyz);
        seed ^= 0xD1B54A32D192ED03ULL;

        const double norm_factor = 3.1415926e-13;
        const double ref_max     = referenceLoop(ref_a.view(norm_factor, relax), ref_a.is_fuel);

        rasbery::XsReconBackend backend;
        if (!backend.available()) {
            std::printf("SKIP backend unavailable: %s\n", backend.status().c_str());
            return 3;
        }
        double dev_max = 0.0;
        if (!backend.solve(dev_a.view(norm_factor, relax), 1, 1, &dev_max)) {
            std::printf("FAIL solve() returned false: %s\n", backend.status().c_str());
            return 4;
        }

        int reported = 0, diffs = 0;
        for (int xt = 0; xt < xsr::NXS; ++xt) {
            char nm[8];
            std::snprintf(nm, sizeof nm, "xs%d", xt);
            diffs += ulpDiffCount(nm, ref_a.xs[xt], dev_a.xs[xt], 10, reported);
        }
        diffs += ulpDiffCount("xs_ssm", ref_a.xs_ssm, dev_a.xs_ssm, 10, reported);
        diffs += ulpDiffCount("iden", ref_a.iden, dev_a.iden, 10, reported);
        if (std::memcmp(&ref_max, &dev_max, sizeof(double)) != 0) {
            std::printf("  DIFF max_change: ref=%.17g got=%.17g\n", ref_max, dev_max);
            ++diffs;
        }
        std::printf("[relax=%.2f] fuel=%d diffs=%d nodes_solved=%llu\n", relax,
                    static_cast<int>(ref_a.fuel.size()), diffs,
                    rasbery::XsReconBackend::nodesSolved());
        total_diffs += diffs;
    }

    if (total_diffs == 0) {
        std::printf("PASS device bit-identical\n");
        return 0;
    }
    std::printf("FAIL %d differing elements\n", total_diffs);
    return 1;
}
