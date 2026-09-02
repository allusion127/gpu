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
// WP15.1.  Both halves of the micro-XS receipt exist here, because Driver.h
// prints the D2D share with no `#ifdef RASBERY_HAS_CUDA` around the call.  A
// CPU-only build uploaded nothing and re-used nothing, so both are zero.
unsigned long long CramBackend::micxD2dBytes() const { return 0; }
unsigned long long CramBackend::bosReuses() const { return 0; }

// WP21-D.  The [RASBERY][CRAM_GPU] receipt calls all four of these
// unconditionally, so the stub has to answer, and the answers are the ones the
// header documents for a mapping that never ran: the serial body, one lane per
// node, no launches, and -1.0 -- "never measured", which reads very differently
// from a measured 0.0 microseconds per launch.
const std::string& CramBackend::kernelVariant() const {
    static const std::string s = "serial";
    return s;
}
int                CramBackend::lanesPerNode() const { return 1; }
unsigned long long CramBackend::launches() const { return 0; }
double             CramBackend::launchUsMean() const { return -1.0; }
// WP20.2.  A CPU-only build ran no pole sum at all, so the honest word is
// the one that says nothing was narrowed.
const char*        CramBackend::poleSumPrecision() const { return "fp64"; }

} // namespace rasbery
