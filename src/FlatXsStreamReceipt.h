#pragma once

// The [RASBERY][FLATXS][STREAM] receipt -- WP23.
//
// WHY A RECEIPT AND NOT A LOG LINE.  Same reason as src/ThGpuReceipt.h, with one
// addition that is specific to this arm.  `RASBERY_GPU_FLATXS_STREAM=1` on a
// deck whose library uses a coordinate form the device body does not implement
// falls back to XSSet::ResolveNodeApplications silently and CORRECTLY -- the
// answers are right, the saving is zero, and the "device" run and the "host" run
// are the same run.  `host_fallback_nodes == nodes` with the flag set is exactly
// that, said out loud.
//
// AND THE FORM CENSUS IS THE GRADE.  Seven of the twenty coordinate forms call
// `log` or `cbrt`; a run that hit one of them is CLASS N1 and no amount of
// contraction mining changes that, while a run that hit none is N1 only because
// this work package did not ship a miner for the three burnup-lerp sites.  Those
// are two different obligations with two different fixes, so the receipt reports
// which one applies rather than printing one grade for both.
//
// PROCESS-WIDE, like every other arm receipt here: in --batch-mode each deck has
// its own XSSet and its own backend, and the question this answers -- did the
// arm fire, how often did it refuse, and by name -- is a question about the run.

#include "FlatXsStreamKernel.h"

#include <atomic>
#include <ios>
#include <ostream>

namespace rasbery::flatxs_stream {

struct StreamTally {
    /// UpdateFlatXS calls taken while the arm was armed.
    std::atomic<unsigned long long> calls{0};
    /// Of those, the ones whose stream the device actually built.
    std::atomic<unsigned long long> device_calls{0};
    /// Of those, the ones a backend refusal handed back to the host builder
    /// WHOLE (not per node): no device, allocation failure, shape refusal.
    std::atomic<unsigned long long> host_calls{0};
    /// Unrodded nodes the arm was asked for, summed over calls.
    std::atomic<unsigned long long> nodes{0};
    /// Of those, the ones the device declined per node and the host rebuilt.
    std::atomic<unsigned long long> host_fallback_nodes{0};
    /// Per-reason breakdown of the line above; index is flatxs_stream::Refusal.
    std::atomic<unsigned long long> refusals[kRefusalCount];
    /// Per-form census, index is the coordinate enumerator.  Counted from the
    /// LIBRARY and the node->model map on the host, not from a device counter:
    /// which forms a node's model carries is static data, and an atomic bump per
    /// term per node would be a receipt that costs more than the arm saves.
    std::atomic<unsigned long long> forms_hit[kFormCount];
    /// Stream entries the device wrote (sum of node_cnt over device nodes).
    std::atomic<unsigned long long> entries{0};
    /// Bytes that did NOT cross the bus because the stream was produced where it
    /// is consumed: the three stream arrays plus node_off/node_cnt that WP13's
    /// uploadGuarded would otherwise have sent.
    std::atomic<unsigned long long> bytes_elided{0};
    /// Bytes that DID cross, so bytes_elided is never read as a total: the
    /// per-node input columns this arm has to upload when RASBERY_GPU_TH is off,
    /// and the node_cnt read-back that carries the refusal ladder.
    std::atomic<unsigned long long> bytes_h2d{0};
    std::atomic<unsigned long long> bytes_d2h{0};
    /// Wall time inside the device entry point, whole call.
    std::atomic<unsigned long long> wall_us{0};
    /// The contraction mask the body was launched with; 0 is a legal mask, so
    /// `forms_seen` says whether one was ever written.
    std::atomic<unsigned long long> forms_mask{0};
    std::atomic<unsigned long long> forms_seen{0};
    /// WHERE THAT MASK CAME FROM, and whether the derivation behind it reached
    /// zero mismatches -- WP23.1.  A value alone cannot distinguish "measured on
    /// this host", "typed by a human" and "fallen back to after the mining
    /// failed", and those three carry different obligations for whoever reads
    /// the receipt.  String literals with static lifetime, written by the
    /// backend at launch; null until then, exactly like `forms_seen`.
    std::atomic<const char*>        forms_source{nullptr};
    std::atomic<unsigned long long> forms_sound{0};
    /// "exact" or "fast" -- which log/cbrt the seven libm forms evaluated.
    std::atomic<const char*>        libm_name{nullptr};
    /// The per-node slot width the arm allocated.  Not a sum: a reader checks
    /// `entries <= nodes * stride`, which a sum could not support.
    std::atomic<unsigned long long> stride{0};
};

inline StreamTally& streamTally() {
    static StreamTally t;
    return t;
}

/// The GRADE, in the string a gate script reads.
///
/// N1 WITH TWO INDEPENDENT REASONS, and the receipt's own fields say which is
/// live on this run.  Neither is an intention dressed as a measurement.
inline constexpr const char* kStreamPolicyNote =
    "RASBERY_GPU_FLATXS_STREAM=1 is CLASS N1, and WP23.1 reduced the reasons from "
    "two to one: (a) the three burnup-lerp contraction sites (FS_REFDENS/"
    "FS_REFDENS0/FS_REFCOND) are now MINED against a verbatim quotation "
    "(src/FlatXsStreamReference.cpp) and forms_source/forms_sound say whether this "
    "run's mask is a measurement of this host -- reason (a) is retired when "
    "forms_sound is 1 and LIVE when it is 0; (b) seven of the twenty coordinate "
    "forms evaluate log/cbrt and this is NOT retired. The exact double-double "
    "path (RASBERY_GPU_FLATXS_STREAM_LIBM=exact) is bit-identical across g++ and "
    "nvcc by construction but still differs from the host, because the host calls "
    "glibc and glibc is the side that is not correctly rounded: 242 of 10^6 "
    "sampled log arguments (max 1 ulp) and 538003 of 10^6 cbrt arguments (max 3 "
    "ulp). So the default is fast, forms_hit says whether this deck touched a libm "
    "form at all, and a libm-free deck with forms_sound=1 is B0-capable "
    "(docs/WP23_FLATXS_STREAM_GPU_20260902_KO.md section 4)";

inline bool streamReceiptWanted() {
    const StreamTally& t = streamTally();
    return t.calls.load(std::memory_order_relaxed) != 0 ||
           t.host_calls.load(std::memory_order_relaxed) != 0;
}

/// True when this run hit at least one form whose coordinate calls libm -- i.e.
/// when reason (b) above is live and not only reason (a).
inline bool streamHitLibmForm() {
    const StreamTally& t = streamTally();
    for (int f = 0; f < kFormCount; ++f)
        if (formUsesLibm(f) && t.forms_hit[f].load(std::memory_order_relaxed) != 0)
            return true;
    return false;
}

/// Write the receipt's fields (no braces, no tag), the way every other receipt
/// in this tree is wrapped by its caller.
inline void appendStreamReceiptFields(std::ostream& os) {
    const StreamTally&       t     = streamTally();
    const unsigned long long calls = t.calls.load(std::memory_order_relaxed);
    const unsigned long long dev   = t.device_calls.load(std::memory_order_relaxed);
    const unsigned long long nodes = t.nodes.load(std::memory_order_relaxed);
    const unsigned long long fb    = t.host_fallback_nodes.load(std::memory_order_relaxed);
    // -1 is the not-measured value, the same convention every other arm uses: a
    // share of 0.0 reads as "refused everything", which is the opposite of
    // "never reached".
    const double share =
        (nodes > 0) ? static_cast<double>(nodes - fb) / static_cast<double>(nodes) : -1.0;

    os << "\"arm\":" << (calls > 0 ? 1 : 0) << ",\"calls\":" << calls
       << ",\"device_calls\":" << dev
       << ",\"host_calls\":" << t.host_calls.load(std::memory_order_relaxed)
       << ",\"nodes\":" << nodes << ",\"host_fallback_nodes\":" << fb
       << ",\"device_node_share\":" << share
       << ",\"entries\":" << t.entries.load(std::memory_order_relaxed)
       << ",\"stride\":" << t.stride.load(std::memory_order_relaxed)
       << ",\"refusals\":{";
    bool first = true;
    for (int r = 1; r < kRefusalCount; ++r) {
        const unsigned long long n = t.refusals[r].load(std::memory_order_relaxed);
        if (n == 0) continue;
        if (!first) os << ',';
        first = false;
        os << '"' << refusalName(r) << "\":" << n;
    }
    os << "},\"forms_hit\":{";
    first = true;
    for (int f = 0; f < kFormCount; ++f) {
        const unsigned long long n = t.forms_hit[f].load(std::memory_order_relaxed);
        if (n == 0) continue;
        if (!first) os << ',';
        first = false;
        os << '"' << formName(f) << "\":" << n;
    }
    os << "},\"libm_form_hit\":" << (streamHitLibmForm() ? 1 : 0)
       << ",\"bytes_elided\":" << t.bytes_elided.load(std::memory_order_relaxed)
       << ",\"bytes_h2d\":" << t.bytes_h2d.load(std::memory_order_relaxed)
       << ",\"bytes_d2h\":" << t.bytes_d2h.load(std::memory_order_relaxed)
       << ",\"wall_ms\":"
       << static_cast<double>(t.wall_us.load(std::memory_order_relaxed)) / 1000.0
       << ",\"forms_mask\":\"";
    if (t.forms_seen.load(std::memory_order_relaxed) > 0)
        os << "0x" << std::hex << t.forms_mask.load(std::memory_order_relaxed) << std::dec;
    else
        os << '~';
    os << "\",\"forms_source\":\"";
    {
        const char* src = t.forms_source.load(std::memory_order_relaxed);
        os << (src != nullptr ? src : "~");
    }
    // -1 IS NOT 0 HERE.  `forms_sound: 0` means the four-seed derivation ran and
    // could not reach a bit-exact mask -- reason (a) is live and this run's
    // rounding contract is unknown.  A run that never launched the arm has no
    // such verdict, and printing 0 for it would report a failure that did not
    // happen.
    os << "\",\"forms_sound\":";
    if (t.forms_seen.load(std::memory_order_relaxed) > 0)
        os << t.forms_sound.load(std::memory_order_relaxed);
    else
        os << -1;
    os << ",\"libm\":\"";
    {
        const char* lm = t.libm_name.load(std::memory_order_relaxed);
        os << (lm != nullptr ? lm : "~");
    }
    os << "\",\"policy_note\":\"" << kStreamPolicyNote << "\"";
}

} // namespace rasbery::flatxs_stream
