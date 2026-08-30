// CPU-only builds: same symbols as CudaPprBackend.cu, no CUDA anywhere, so the
// call site in PPR.cpp never needs an #ifdef.  Mirrors CudaXsReconBackendStub.cpp.

#include "CudaPprBackend.h"

namespace rasbery {

struct PprBackend::Impl {
    /// WP6 stage F.  The refusal ladder is real in a CPU-only build too:
    /// PPR::resetAndDriveGpu calls noteHostFallback on this object, so the
    /// receipt says `arm_off:35` rather than leaving the reader to infer it.
    ppr::Refusal       last_refusal = ppr::Refusal::None;
    unsigned long long refusals[static_cast<int>(ppr::Refusal::Count)] = {};
};

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

// WP6 stage F.  THE LADDER EXISTS EVEN HERE, and it has exactly one rung: a
// stub build's arm is off, for every statepoint, and saying so is the whole
// point of the ladder.  The counter is real -- PPR::resetAndDriveGpu calls
// noteHostFallback on this object in a CPU-only build too, so `refusals`
// answers "how many statepoints" rather than "was CUDA compiled in".
void PprBackend::noteHostFallback(ppr::Refusal reason) {
    const int i = static_cast<int>(reason);
    if (i < 0 || i >= static_cast<int>(ppr::Refusal::Count)) return;
    _impl->last_refusal = reason;
    _impl->refusals[i] += 1;
}

ppr::Refusal PprBackend::lastRefusal() const { return _impl->last_refusal; }

const char* PprBackend::lastRefusalName() const {
    return ppr::refusalName(_impl->last_refusal);
}

unsigned long long PprBackend::refusalCount(ppr::Refusal reason) const {
    const int i = static_cast<int>(reason);
    if (i < 0 || i >= static_cast<int>(ppr::Refusal::Count)) return 0;
    return _impl->refusals[i];
}

std::string PprBackend::refusalJson() const {
    std::string out   = "{";
    bool        first = true;
    for (int i = 1; i < static_cast<int>(ppr::Refusal::Count); ++i) {
        if (_impl->refusals[i] == 0) continue;
        if (!first) out += ",";
        first = false;
        out += "\"";
        out += ppr::refusalName(static_cast<ppr::Refusal>(i));
        out += "\":";
        out += std::to_string(_impl->refusals[i]);
    }
    out += "}";
    return out;
}

} // namespace rasbery
