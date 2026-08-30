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

bool PprBackend::reconstructPinPower(const ppr::ReconGeomView&, const ppr::ReconStepView&) {
    return false;
}

unsigned long long PprBackend::statepoints() const { return 0; }
unsigned long long PprBackend::iterations() const { return 0; }
double             PprBackend::wallMs() const { return 0.0; }
int                PprBackend::deviceOrdinal() const { return -1; }

// WP6 receipt counters.  A stub build never runs the arm, so every one of them
// is zero and `loopArm` names the reason rather than a loop that does not
// exist -- the receipt is printed under RASBERY_STATEPOINT_TELEMETRY even here.
const char*        PprBackend::loopArm() const { return "none"; }
unsigned long long PprBackend::hostSyncs() const { return 0; }
double             PprBackend::hostSyncsPerStatepoint() const { return 0.0; }
unsigned long long PprBackend::graphLaunches() const { return 0; }
unsigned long long PprBackend::graphBuilds() const { return 0; }
const std::string& PprBackend::graphRefusal() const {
    static const std::string s = "built without CUDA (stub)";
    return s;
}
unsigned long long PprBackend::h2dBytes() const { return 0; }
unsigned long long PprBackend::h2dBytesElided() const { return 0; }
unsigned long long PprBackend::d2hBytes() const { return 0; }
unsigned long long PprBackend::canonicalStatepoints() const { return 0; }
unsigned long long PprBackend::canonicalMismatch() const { return 0; }
unsigned long long PprBackend::allocations() const { return 0; }
unsigned long long PprBackend::reallocations() const { return 0; }
void               PprBackend::noteReconRepair() {}
unsigned long long PprBackend::reconRepairs() const { return 0; }
unsigned long long PprBackend::reconStatepoints() const { return 0; }
unsigned long long PprBackend::pinMaterializations() const { return 0; }
const std::string& PprBackend::reconRefusal() const {
    static const std::string s = "built without CUDA (stub)";
    return s;
}

} // namespace rasbery
