// CPU-only builds: same symbols as CudaPprBackend.cu, no CUDA anywhere, so the
// call site in PPR.cpp never needs an #ifdef.  Mirrors CudaXsReconBackendStub.cpp.

#include "CudaPprBackend.h"

namespace rasbery {

struct PprBackend::Impl {};

PprBackend::PprBackend() : _impl(new Impl) {}
PprBackend::~PprBackend() = default;

bool PprBackend::available() const { return false; }

const std::string& PprBackend::status() const {
    static const std::string s = "built without CUDA (stub)";
    return s;
}

bool PprBackend::resetAndDrive(const ppr::GeomView&, const ppr::StepView&, int, int*) {
    return false;
}

unsigned long long PprBackend::statepoints() const { return 0; }
unsigned long long PprBackend::iterations() const { return 0; }
double             PprBackend::wallMs() const { return 0.0; }
int                PprBackend::deviceOrdinal() const { return -1; }

} // namespace rasbery
