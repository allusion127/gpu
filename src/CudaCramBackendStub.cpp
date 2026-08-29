// CPU-only builds: same symbols as CudaCramBackend.cu, no CUDA anywhere, so the
// call sites in XSSet.cpp never need an #ifdef.  Mirrors CudaPprBackendStub.cpp.

#include "CudaCramBackend.h"

namespace rasbery {

struct CramBackend::Impl {};

CramBackend::CramBackend() : _impl(new Impl) {}
CramBackend::~CramBackend() = default;

bool CramBackend::available() const { return false; }

const std::string& CramBackend::status() const {
    static const std::string s = "built without CUDA (stub)";
    return s;
}

bool CramBackend::predictor(const cram::LibView&, const cram::PredictorView&,
                            unsigned long long*) {
    return false;
}

bool CramBackend::corrector(const cram::LibView&, const cram::CorrectorView&) {
    return false;
}

unsigned long long CramBackend::predictorCalls() const { return 0; }
unsigned long long CramBackend::correctorCalls() const { return 0; }
unsigned long long CramBackend::nodesSolved() const { return 0; }
unsigned long long CramBackend::gsIterations() const { return 0; }
unsigned long long CramBackend::gsSolves() const { return 0; }
double             CramBackend::wallMs() const { return 0.0; }
int                CramBackend::deviceOrdinal() const { return -1; }
unsigned long long CramBackend::micxH2dBytes() const { return 0; }
unsigned long long CramBackend::bosReuses() const { return 0; }

} // namespace rasbery
