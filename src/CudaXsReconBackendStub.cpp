// CPU-only builds: same symbols as CudaXsReconBackend.cu, no CUDA anywhere,
// so call sites never need an #ifdef.  Mirrors CudaBICGBackendStub.cpp and
// CudaDepletionBackendStub.cpp.

#include "CudaXsReconBackend.h"

namespace rasbery {

struct XsReconBackend::Impl {};

XsReconBackend::XsReconBackend()  = default;
XsReconBackend::~XsReconBackend() = default;

bool XsReconBackend::available() const { return false; }

const std::string& XsReconBackend::status() const {
    static const std::string s = "built without CUDA (stub)";
    return s;
}

bool XsReconBackend::solve(const xsrecon::BatchView&, unsigned long long,
                           unsigned long long, double*) {
    return false;
}

bool XsReconBackend::solveFlatXs(const flatxs::FlatXsView&, const FlatXsLibShape&,
                                 unsigned long long, unsigned long long,
                                 unsigned long long, unsigned long long, bool) {
    return false;
}

// Rev.7.1 Task 13 stub parity: the split Xe arm fails open, which is exactly
// what a no-CUDA build should do -- every one of these returning false means
// the caller runs the host Anderson/Picard path, unchanged.
bool XsReconBackend::xeEvaluate(const xsrecon::BatchView&, unsigned long long,
                                unsigned long long, double*) {
    return false;
}

bool XsReconBackend::xeRotateHistory() { return false; }

bool XsReconBackend::xeRecordColumn(int) { return false; }

bool XsReconBackend::xeSaveEvaluation() { return false; }

bool XsReconBackend::xeDots(int, double*) { return false; }

bool XsReconBackend::xeCandidate(const double*, int, double*, bool*) { return false; }

bool XsReconBackend::xeCommit(const xsrecon::BatchView&, int, double, bool,
                              unsigned long long) {
    return false;
}

// WP7-C.  The transaction declines here for the same reason the six above do:
// a call site must never need an #ifdef.
bool XsReconBackend::xeTransaction(const xsrecon::BatchView&, unsigned long long,
                                   unsigned long long, const XeTxnRequest&,
                                   xe::XeTxnControl*) {
    return false;
}

unsigned long long XsReconBackend::xeEvaluations() { return 0; }

unsigned long long XsReconBackend::xeCommits() { return 0; }

// WP15.  A stub build has no device block, so nothing is ever owed: the two
// predicates are false, the payer is a no-op that succeeds (its callers treat
// false as "the device could not hand the block back", which is not the case
// here), and every receipt is zero.
bool XsReconBackend::micxScalarsPending() const { return false; }

bool XsReconBackend::micxScatterPending() const { return false; }

unsigned long long XsReconBackend::micxResidentGeneration() const { return 0; }

const void* XsReconBackend::micxDeviceSlot(int) const { return nullptr; }
int XsReconBackend::micxDeviceElemBytes() const { return static_cast<int>(sizeof(double)); }

void* XsReconBackend::micxReadyEvent() { return nullptr; }

unsigned long long XsReconBackend::nodalJnetElidedBytes() { return 0; }

unsigned long long XsReconBackend::nodalJnetElisionHits() { return 0; }

unsigned long long XsReconBackend::nodalJnetElisionTests() { return 0; }

bool XsReconBackend::downloadFlatXsMicx(const flatxs::FlatXsView&, bool, bool) {
    return true;
}

unsigned long long XsReconBackend::micxResidentHits() { return 0; }

unsigned long long XsReconBackend::micxLazyDownloads() { return 0; }

unsigned long long XsReconBackend::micxSliceDownloads() { return 0; }

unsigned long long XsReconBackend::micxBytesSaved() { return 0; }

unsigned long long XsReconBackend::nodalConstUploads() { return 0; }

unsigned long long XsReconBackend::nodalConstBytes() { return 0; }

unsigned long long XsReconBackend::nodesSolved() { return 0; }

unsigned long long XsReconBackend::flatXsNodesSolved() { return 0; }

bool XsReconBackend::solveNodal(const nodal::NodalView&, unsigned long long,
                                unsigned long long, unsigned long long) {
    return false;
}

bool XsReconBackend::solveNodalPost(const nodal::NodalView&) { return false; }

unsigned long long XsReconBackend::nodalDrivesSolved() { return 0; }

// Lease bookkeeping without a device call: no hook is installed in a stub
// build, so the registry tracks ownership and the owner destructors exercise
// the same release path they take under CUDA.
bool XsReconBackend::pinHost(const void* p, size_t bytes, const char* tag) {
    return rasberyPinHost(p, bytes, tag);
}

// Rev.7.1 Task 7 stub parity.  A no-CUDA build has no device buffers to borrow,
// so adoption is accepted and ignored and every accessor answers "legacy".  The
// call sites therefore need no #ifdef, which is the same contract every other
// symbol in this file keeps.
void XsReconBackend::adoptCanonicalBuffers(const gpu::CanonicalSlotBuffers&) {}
/// No device, no batched nodal arena, nothing to dishonour.  True keeps the
/// caller's branch the same shape in both builds; the adoption it guards can
/// never happen here anyway, because the set is always empty.

gpu::CanonicalSlotBuffers XsReconBackend::canonicalBuffers() const {
    return gpu::CanonicalSlotBuffers{};
}

void XsReconBackend::setMaterializeMask(std::uint32_t) {}

void XsReconBackend::setCanonicalNodalSegmentMode(bool, bool) {}

std::uint32_t XsReconBackend::materializeMask() const { return 0u; }

void*              XsReconBackend::nodalCompletionEvent() { return nullptr; }
void*              XsReconBackend::nodalReigvDeviceSlot() const { return nullptr; }
void               XsReconBackend::setNodalReigvDeviceResident(bool) {}
unsigned long long XsReconBackend::canonicalUploadsElided() const { return 0; }

unsigned long long XsReconBackend::canonicalDownloadsElided() const { return 0; }

bool rasberyGpuXsReconEnabled() { return false; }

bool rasberyGpuFlatXsEnabled() { return false; }

bool rasberyGpuMicxResidentEnabled() { return false; }

bool rasberyGpuFlatXsCtaEnabled() { return false; }

// The ladder's default, not 0: a caller that logs the value must not be told
// the arm would launch an empty block.  With the CUDA-less build there is no
// arm at all, and rasberyGpuFlatXsCtaEnabled() already says so.
int rasberyGpuFlatXsCtaThreads() { return 128; }

// WP21-B2.  0 = "the arm's own default", and there is no arm here; the three
// receipt counters read zero for the same reason nodes_solved does.
int                rasberyGpuFlatXsCtaTile() { return 0; }
int                rasberyGpuFlatXsCtaTileRan() { return 0; }
unsigned long long rasberyGpuFlatXsCtaTilesLaunched() { return 0; }
unsigned long long rasberyGpuFlatXsCtaTailNodes() { return 0; }

bool rasberyGpuNodalEnabled() { return false; }

// WP21-C2.  No device, no device array, no permutation.
bool rasberyGpuNodalSoaEnabled() { return false; }

bool rasberyGpuXeEnabled() { return false; }

bool rasberyGpuXeTxnEnabled() { return false; }

int rasberyGpuXeDotPartitions() { return 0; }

unsigned long long rasberyGpuXeEvaluations() { return 0; }

unsigned long long rasberyGpuXeCommits() { return 0; }

unsigned long long rasberyGpuNodalDrives() { return 0; }

unsigned long long rasberyGpuXsReconNodes() { return 0; }

unsigned long long rasberyGpuFlatXsNodes() { return 0; }

unsigned long long rasberyGpuNodalCanonicalElidedUploadBytes() { return 0; }
unsigned long long rasberyGpuNodalCanonicalElidedDownloadBytes() { return 0; }

} // namespace rasbery
