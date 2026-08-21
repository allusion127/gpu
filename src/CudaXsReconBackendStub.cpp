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

void XsReconBackend::pinHost(const void*, size_t) {}

bool rasberyGpuXsReconEnabled() { return false; }

bool rasberyGpuFlatXsEnabled() { return false; }

unsigned long long rasberyGpuXsReconNodes() { return 0; }

unsigned long long rasberyGpuFlatXsNodes() { return 0; }

} // namespace rasbery
