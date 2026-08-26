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

bool rasberyGpuXsReconEnabled() { return false; }

bool rasberyGpuFlatXsEnabled() { return false; }

bool rasberyGpuNodalEnabled() { return false; }

unsigned long long rasberyGpuNodalDrives() { return 0; }

unsigned long long rasberyGpuXsReconNodes() { return 0; }

unsigned long long rasberyGpuFlatXsNodes() { return 0; }

} // namespace rasbery
