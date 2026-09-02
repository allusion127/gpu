#pragma once

// The critical-boron search's device half -- WP22 commit 2.
//
// ---------------------------------------------------------------------------
// WHAT MOVED, AND -- MORE IMPORTANTLY -- WHAT DID NOT
// ---------------------------------------------------------------------------
//
// A boron trial is three things:
//
//   1. THE SECANT.  Two scalars in, one scalar out (Scheduler.h
//      ProposeNextSearchPoint / CommitSearchPoint).  It STAYS ON THE HOST, and
//      that is a decision rather than an omission: it is O(1) arithmetic on a
//      k_eff the solve already published to the host, so moving it would buy
//      nanoseconds and cost a kernel launch, a scalar download and a second
//      place where the bracket logic lives.  What the arm owes here is the
//      NEGATIVE property -- that the propose step transfers nothing -- and
//      tools/test_search_gpu_contract.py asserts it against the source.
//
//   2. THE PER-NODE BORON WRITE.  `for l: bppm[l] = x` (XSSet::SetBoron).  This
//      is what moves: the same broadcast, written by a device kernel into the
//      flat-XS backend's RESIDENT per-node block, so the trial's boron reaches
//      the reconstruction without an nxyz-double host-to-device copy.
//
//   3. THE RECONSTRUCTION ITSELF, which was already on the device
//      (RASBERY_GPU_FLATXS) and which this arm does not touch.
//
// THE HOST MIRROR IS STILL WRITTEN, and pretending otherwise would be the one
// mistake this file could make.  XSSet::BuildFlatXsStream resolves the branch
// stream on the host and reads `_g.bppm(l)` per node to form the boron
// coordinate, so the host array is a live input, not a stale copy.  The arm
// removes a TRANSFER, not a host array; docs/WP22_TH_SEARCH_GPU_20260902_KO.md
// names moving BuildFlatXsStream as the next lever and prices it.
//
// ---------------------------------------------------------------------------
// CLASS B0, BY CONSTRUCTION AND NOT BY MEASUREMENT
// ---------------------------------------------------------------------------
//
// There is no arithmetic here.  Every node receives THE SAME double the host
// secant proposed -- a store, not a computation -- so there is no rounding
// decision, no contraction to mine, and no form mask.  The bytes the device
// block holds after the kernel are the bytes the H2D would have written, which
// is the same argument rasbery::xfer's elision arm rests on (XferLedger.h: "the
// device buffer's contents after the elided copy are the same bits as after the
// copy, so no kernel can observe the difference").
//
// That is why this header is four functions long and ThKernel.h is four hundred
// lines: the difference between the two arms is not effort, it is that one of
// them moves arithmetic and the other moves bytes.

#include <cstddef>

#if defined(__CUDACC__)
    #define RASBERY_SEARCH_HD __host__ __device__
#else
    #define RASBERY_SEARCH_HD
#endif

namespace rasbery::search {

/// One node of XSSet::SetBoron's broadcast.  Deliberately a function rather
/// than an inline `dst[l] = value` at the call site: the contract test can then
/// assert that the device kernel and the host loop write through THE SAME body,
/// which is the only form in which "the device wrote what the host would have"
/// is checkable rather than asserted.
RASBERY_SEARCH_HD inline void searchBoronBroadcastNode(double* bppm, int l, double value) {
    bppm[l] = value;
}

/// What one boron apply is: the value and the shape.  No pointers -- the device
/// block is the flat-XS backend's own and is addressed there, so this view
/// cannot be handed a buffer that does not belong to the arm.
struct BoronApplyView {
    double value = 0.0;
    int    nxyz  = 0;
};

/// Is this apply one the device arm can serve?  A shape mismatch DECLINES; a
/// kernel that wrote a prefix of the block would leave the tail holding the
/// previous trial's boron, which is a plausible core and a wrong one.
RASBERY_SEARCH_HD inline bool searchBoronApplyServable(const BoronApplyView& v,
                                                       int resident_nxyz) {
    return v.nxyz > 0 && v.nxyz == resident_nxyz;
}

/// The host mirror's own write, so the two spellings are one text.  Called by
/// XSSet::SetBoron on both arms -- the device arm does NOT remove this loop,
/// because XSSet::BuildFlatXsStream reads the host array per node.
inline void searchBoronBroadcastHost(double* bppm, int nxyz, double value) {
    for (int l = 0; l < nxyz; ++l) searchBoronBroadcastNode(bppm, l, value);
}

} // namespace rasbery::search
