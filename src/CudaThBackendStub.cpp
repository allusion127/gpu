// CPU-only builds: same symbols as CudaThBackend.cu, no CUDA anywhere, so the
// call sites in XSSet.cpp never need an #ifdef.  Mirrors CudaCramBackendStub.cpp.
//
// EVERY ENTRY POINT RETURNS false, which is what makes the feature-off path
// byte-identical in a build without a device: XSSet::UpdateTH sees the decline
// and runs SolveTH unchanged.

#include "CudaThBackend.h"

namespace rasbery {

struct ThBackend::Impl {};

ThBackend::ThBackend() : _impl(new Impl) {}
ThBackend::~ThBackend() = default;

bool ThBackend::available() const { return false; }

const std::string& ThBackend::status() const {
    static const std::string s = "built without CUDA (stub)";
    return s;
}

bool ThBackend::solveTh(const thgpu::TableView&, const thgpu::GeomView&,
                        const thgpu::UpdateView&, double&) {
    return false;
}

unsigned long long ThBackend::deviceUpdates() const { return 0; }
unsigned long long ThBackend::bytesElided() const { return 0; }
unsigned long long ThBackend::bytesH2d() const { return 0; }
unsigned long long ThBackend::bytesD2h() const { return 0; }
double             ThBackend::wallMs() const { return 0.0; }
int                ThBackend::deviceOrdinal() const { return -1; }
unsigned long long ThBackend::formsMask() const { return 0; }

} // namespace rasbery
