// Consistency harness for XsReconKernel.h.
//
// Gate: the shared host/device node function must reproduce the CPU loop of
// XSSet::UpdateEquilibriumXenon BIT-IDENTICALLY on synthetic data -- exact
// zero difference, no tolerance.  The reconstruction is a finite ascending
// sum, so unlike the depletion probe's iterative solver there is no residual
// ball to hide in; any deviation is a transcription or contraction bug.
//
// The reference below is quoted from src/XSSet.cpp (UpdateEquilibriumXenon's
// fuel-node loop + ApplyXeEquilibrium + ReconstructNode), with member access
// rewritten to locals.  Drift between that file and this quotation would
// invalidate the harness, so it quotes rather than paraphrases -- same rule as
// the depletion consistency harness.
//
// What this proves: the shared function's arithmetic ORDER equals the CPU
// loop's under one compiler.  What it cannot prove: that nvcc's floating
// point equals g++'s for the same source (contraction).  That last step is
// closed on the GPU host by the flag-on/flag-off A/B of the real binary,
// whose receipts compare bit-for-bit (tools/xsrecon_gpu_ab.py).
//
// Usage: rasbery_xsrecon_consistency [nxyz]     (default 4096)

#include "xsrecon_test_common.h"

int main(int argc, char** argv) {
    const int nxyz = (argc > 1) ? std::atoi(argv[1]) : 4096;
    if (nxyz <= 0) {
        std::printf("bad nxyz\n");
        return 2;
    }

    int total_diffs = 0;
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
    for (double relax : {1.0, 0.5}) {
        // Both arms must see byte-identical inputs: re-seed before EACH build.
        g_state      = seed;
        Arrays ref_a = makeArrays(nxyz);
        g_state      = seed;
        Arrays ker_a = makeArrays(nxyz);
        seed ^= 0xD1B54A32D192ED03ULL; // fresh inputs for the next arm

        const double norm_factor = 3.1415926e-13;
        const double ref_max     = referenceLoop(ref_a.view(norm_factor, relax), ref_a.is_fuel);

        xsr::BatchView kv      = ker_a.view(norm_factor, relax);
        double         ker_max = 0.0;
        for (int i = 0; i < kv.n_fuel; ++i) {
            double mc = 0.0;
            if (xsreconSolveNode(kv, kv.fuel[i], &mc))
                ker_max = std::max(ker_max, mc);
        }

        int reported = 0, diffs = 0;
        for (int xt = 0; xt < xsr::NXS; ++xt) {
            char nm[8];
            std::snprintf(nm, sizeof nm, "xs%d", xt);
            diffs += ulpDiffCount(nm, ref_a.xs[xt], ker_a.xs[xt], 10, reported);
        }
        diffs += ulpDiffCount("xs_ssm", ref_a.xs_ssm, ker_a.xs_ssm, 10, reported);
        diffs += ulpDiffCount("iden", ref_a.iden, ker_a.iden, 10, reported);
        if (std::memcmp(&ref_max, &ker_max, sizeof(double)) != 0) {
            std::printf("  DIFF max_change: ref=%.17g got=%.17g\n", ref_max, ker_max);
            ++diffs;
        }
        std::printf("[relax=%.2f] fuel=%d diffs=%d max_change=%.17g\n", relax,
                    kv.n_fuel, diffs, ker_max);
        total_diffs += diffs;
    }

    if (total_diffs == 0) {
        std::printf("PASS bit-identical\n");
        return 0;
    }
    std::printf("FAIL %d differing elements\n", total_diffs);
    return 1;
}
