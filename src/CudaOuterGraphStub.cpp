// CPU-only builds: the no-CUDA arm of the device outer segment (Rev.7.1 Task 9).
// Mirrors CudaBICGBackendStub.cpp and CudaXsReconBackendStub.cpp -- every symbol
// CudaOuterGraph.cu defines has a definition here, and every one of them answers
// "no device" rather than failing to link.
//
// ITS OWN TRANSLATION UNIT, DELIBERATELY.  The obvious home was
// GpuPhysicsBackendStub.cpp, next to the scheduler launchers it sits beside in
// the CUDA build.  Putting it there drags CudaOuterGraph.h -- and through it
// CmfdOuterKernel.h and GpuFormMask.h, which calls std::getenv -- into a file
// the Task 1 interface gate compiles standalone at /W4 /WX, where getenv is
// C4996 and therefore an error.  A per-backend stub file is also what every
// other backend in this tree already does, so this is the convention rather than
// a workaround.
//
// WHAT IS *NOT* HERE, and why.  deviceOuterTransition(), outerApplyTransition()
// and outerDeckHasFractionalRods() are pure and live in the header, so they run
// on the host in any build -- which is what lets test/outer_state_replay.cpp
// check the escape ranking, the budget precedence and the kPhaseTransitions
// consistency with no device at all.  Only the RUNNER refuses here.
//
// The gates and the receipt are REAL, not stubbed to zero: a CPU-only build
// asked for RASBERY_GPU_OUTER=1 must still print the line and say
// `refusals:{no_runner:...}`, because "the feature was on and never engaged" is
// exactly the state the receipt exists to expose.

#include "CudaOuterGraph.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <string>

namespace rasbery::gpu {

namespace {

bool outerEnvFlagOn(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    const std::string s(v);
    return !(s.empty() || s == "0" || s == "off" || s == "OFF" || s == "false" ||
             s == "FALSE");
}

std::uint64_t& stubRefusals(OuterSegmentRefusal r) {
    static std::uint64_t counts[static_cast<int>(OuterSegmentRefusal::Count)] = {};
    return counts[static_cast<int>(r)];
}

} // namespace

bool outerGpuEnabled() {
    static const bool on = outerEnvFlagOn("RASBERY_GPU_OUTER");
    return on;
}

unsigned int outerSegmentBudget() {
    static const unsigned int budget = [] {
        const char* v = std::getenv("RASBERY_GPU_OUTER_SEGMENT_MAX");
        if (v == nullptr) return kOuterSegmentBudgetDefault;
        const long parsed = std::atol(v);
        // The SAME diagnostic as the CUDA arm, deliberately.  A run that asked
        // for a budget it did not get must say so in EITHER build: the receipt
        // reports `segment_budget`, but a reader who set 999 and sees 8 is owed
        // the reason at the point the value was refused, not an inference.
        if (parsed < 1 || parsed > static_cast<long>(kOuterSegmentBudgetMax)) {
            std::fprintf(stderr,
                         "[RASBERY][OUTER_GPU][WARN] RASBERY_GPU_OUTER_SEGMENT_MAX=%s is "
                         "outside [1, %u]; using the default %u\n",
                         v, kOuterSegmentBudgetMax, kOuterSegmentBudgetDefault);
            return kOuterSegmentBudgetDefault;
        }
        return static_cast<unsigned int>(parsed);
    }();
    return budget;
}

OuterSegmentCounters outerSegmentCounters() {
    OuterSegmentCounters out;
    for (int i = 0; i < static_cast<int>(OuterSegmentRefusal::Count); ++i)
        out.refusals[i] = stubRefusals(static_cast<OuterSegmentRefusal>(i));
    return out;
}

std::string outerSegmentReceiptJson() {
    const OuterSegmentCounters c = outerSegmentCounters();
    std::string s = "{\"segment_launches\":0,\"device_outers\":0,"
                    "\"host_outer_observations\":0,\"budget_exits\":0,"
                    "\"halted_outer_launches\":0,\"segment_budget\":" +
                    std::to_string(outerSegmentBudget()) + ",\"escapes\":{},\"refusals\":{";
    bool first = true;
    for (int i = 0; i < static_cast<int>(OuterSegmentRefusal::Count); ++i) {
        if (c.refusals[i] == 0) continue;
        if (!first) s += ",";
        first = false;
        s += "\"";
        s += outerRefusalName(static_cast<OuterSegmentRefusal>(i));
        s += "\":" + std::to_string(c.refusals[i]);
    }
    s += "}}";
    return s;
}

void reportOuterSegment(std::ostream& os) {
    if (!outerGpuEnabled()) return;
    os << "[RASBERY][OUTER_GPU] " << outerSegmentReceiptJson() << std::endl;
}

void noteOuterSegmentRefusal(OuterSegmentRefusal why) {
    if (why == OuterSegmentRefusal::None) return;
    ++stubRefusals(why);
}

struct CudaOuterSegment::Impl {
    std::string         status = "no CUDA in this build";
    OuterSegmentHooks   hooks{};
    OuterSegmentBinding binding{};
};

CudaOuterSegment::CudaOuterSegment() : _impl(new Impl) {}
CudaOuterSegment::~CudaOuterSegment() { delete _impl; }

bool CudaOuterSegment::initialize(const DeviceArenaView&, int) { return false; }
void CudaOuterSegment::release() {}
bool CudaOuterSegment::available() const { return false; }
const std::string& CudaOuterSegment::status() const { return _impl->status; }
void CudaOuterSegment::bind(const OuterSegmentBinding& binding) { _impl->binding = binding; }
bool CudaOuterSegment::bound() const { return false; }
void CudaOuterSegment::setHooks(const OuterSegmentHooks& hooks) { _impl->hooks = hooks; }
OuterSegmentHooks CudaOuterSegment::hooks() const { return _impl->hooks; }

OuterSegmentRefusal CudaOuterSegment::refusal(int, bool, bool) const {
    return outerGpuEnabled() ? OuterSegmentRefusal::NoRunner : OuterSegmentRefusal::FeatureOff;
}

bool CudaOuterSegment::runSegment(const OuterSegmentScalars&, int batch_width,
                                  bool fractional_rods, bool critical_search,
                                  OuterSegmentResume&) {
    ++stubRefusals(refusal(batch_width, fractional_rods, critical_search));
    return false;
}

} // namespace rasbery::gpu
