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

// Forward-declared rather than included: the receipt needs the batch width to name
// the idle reason, and CudaBICGBackend.h would drag the whole solver surface into a
// file that only wants one int.  The declaration matches CudaBICGBackend.h:336.
namespace rasbery {
int rasberyBatchWidth();
} // namespace rasbery

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
                    "\"halted_outer_launches\":0,\"jnet_bridge_bytes\":0,"
                    "\"flux_sync_bytes\":0,\"host_mirror_bytes\":0,"
                    "\"cusping_fired\":0,\"cusping_dtil_bytes\":0,"
                    "\"device_flux_outers\":0,\"flux_uploads_elided\":0,"
                    "\"xsnf_uploads_elided\":0,\"dtil_uploads_elided\":0,"
                    "\"mirror_exits\":0,\"canonical_nodal_outers\":0,"
                    "\"phis_mirror_bytes\":0,\"jnet_mirror_bytes\":0,"
                    "\"segment_budget\":" +
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
    s += "},";
    // Printed ONLY when nothing ran: on a healthy run the reason is "none" and
    // saying so would be noise, while on an idle run it is the whole message.
    if (c.segment_launches == 0) {
        s += outerIdleReasonJson(
            rasberyOuterSegment().refusal(rasberyBatchWidth(), false, false, true));
        s += ",";
    }
    s += outerHostBodyJson();
    s += "}";
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

bool CudaOuterSegment::initialize(const DeviceArenaView&, int, int) { return false; }
void CudaOuterSegment::release() {}
int  CudaOuterSegment::slot() const { return -1; }
bool CudaOuterSegment::available() const { return false; }
const std::string& CudaOuterSegment::status() const { return _impl->status; }
void CudaOuterSegment::bind(const OuterSegmentBinding& binding) { _impl->binding = binding; }
bool CudaOuterSegment::bound() const { return false; }
void CudaOuterSegment::setHooks(const OuterSegmentHooks& hooks) { _impl->hooks = hooks; }
OuterSegmentHooks CudaOuterSegment::hooks() const { return _impl->hooks; }
bool CudaOuterSegment::bindResidency(const OuterSegmentResidency&) { return false; }
bool CudaOuterSegment::residencyBound() const { return false; }
/// Rev.7.1 Task 18-lite.  Nothing to adopt without a device, so the set is
/// empty and the binding can never be live -- which is what makes the runner's
/// `bridge or not` test answer `bridge` in a stub build without a second path.
CanonicalSlotBuffers CudaOuterSegment::canonicalNodalSet() const {
    return CanonicalSlotBuffers{};
}
void CudaOuterSegment::setCanonicalNodalBound(bool) { _impl->binding.canonical_nodal = false; }
bool CudaOuterSegment::canonicalNodalBound() const { return false; }
bool CudaOuterSegment::publishProbe(int, double, double, bool, bool) { return false; }
/// Rev.7.1 Task 10 part 2.  No stream to adopt, no device probe to address and
/// nothing to unlatch: the runner is unavailable, so every one of these answers
/// the same "not here" the rest of this file does.
bool CudaOuterSegment::useStream(void*) { return false; }
CudaOuterSegment::ProbeAddresses CudaOuterSegment::probeAddresses(int) const {
    return ProbeAddresses{};
}
bool CudaOuterSegment::republishAfterHostSweep(int, double, double, bool, bool) {
    return false;
}

OuterSegmentRefusal CudaOuterSegment::refusal(int, bool, bool, bool) const {
    return outerGpuEnabled() ? OuterSegmentRefusal::NoRunner : OuterSegmentRefusal::FeatureOff;
}

bool CudaOuterSegment::runSegment(const OuterSegmentScalars& scalars, int batch_width,
                                  bool fractional_rods, bool critical_search,
                                  OuterSegmentResume&) {
    ++stubRefusals(
        refusal(batch_width, fractional_rods, critical_search, scalars.slot_admitted != 0));
    return false;
}

// ---------------------------------------------------------------------------
// Rev.7.1 Task 18-lite: the per-slot table, with no slots to serve
// ---------------------------------------------------------------------------
//
// ONE OBJECT, NOT kMaxDeviceSlots OF THEM.  The CUDA arm needs a distinct runner
// per slot because each holds a distinct residency; this arm holds none, refuses
// every call, and would be paying for 64 identical refusals.  The index is
// accepted and discarded, which is the same shape every other function here has.

CudaOuterSegment& rasberyOuterSegment(int) {
    static CudaOuterSegment segment;
    return segment;
}

/// No arena was stood up, so it has no width.  The ladder never reaches the
/// batch test in this build -- `no_runner` is ranked above it -- so this is
/// reported for the receipt's sake and nothing branches on it.
int rasberyOuterArenaSlots() { return 0; }

bool rasberyOuterSlotAdmitted(int, const OuterSegmentDeckShape&) { return false; }

/// The counters are process-wide here (stubRefusals), so there is no per-slot
/// set to aim a thread at.
void outerSetThreadSlot(int) {}


// ---------------------------------------------------------------------------
// Standing the segment up -- the no-CUDA answer  (Task 9, link 1)
// ---------------------------------------------------------------------------
//
// IT STILL COUNTS THE REFUSAL, and that is the whole point of having a stub
// here rather than an #ifdef at the call site.  A CPU-only build asked for
// RASBERY_GPU_OUTER=1 must print the same receipt with the same idle reason as
// a CUDA build that found no device -- "the feature was on and never engaged"
// is one state, not two, and Driver.h should not have to know which build it is
// compiled into to say so.

bool rasberyStandUpOuterSegment(const OuterSegmentDeck&, std::ostream&) {
    if (!outerGpuEnabled()) return false;
    ++stubRefusals(OuterSegmentRefusal::NoRunner);
    std::fprintf(stderr, "[RASBERY][OUTER_GPU][WARN] RASBERY_GPU_OUTER=1 in a build with no "
                         "CUDA; the device outer stays off\n");
    return false;
}

void rasberyTearDownOuterSegment() {}

// Link 2 has no meaning without a device: there are no arena addresses to hand
// over and no probe to publish.  Both answer false so the caller takes the same
// path it takes when a CUDA build has no arena -- one refusal, one name.
bool rasberyBindOuterResidency(const OuterSegmentResidency&) { return false; }

bool rasberyPublishOuterProbe(int, double, double, bool, bool) { return false; }

} // namespace rasbery::gpu
