#include "CudaBICGBackend.h"

#include "Geometry.h"
#include "HostPinRegistry.h"

#include <stdexcept>

namespace rasbery {

class CudaBICGBackend::Impl {
public:
    std::string status = "CUDA support was not compiled";
};

CudaBICGBackend::CudaBICGBackend(Geometry&) : _impl(std::make_unique<Impl>()) {}
CudaBICGBackend::~CudaBICGBackend() = default;

bool CudaBICGBackend::available() const { return false; }
const std::string& CudaBICGBackend::status() const { return _impl->status; }
BackendCounters CudaBICGBackend::counters() const { return {}; }
DeviceSolveStatus CudaBICGBackend::lastSolveStatus() const { return {}; }

void CudaBICGBackend::reset(const double*, const double*, const double*, const double*) {
    throw std::runtime_error(_impl->status);
}

void CudaBICGBackend::solveInner(int, double) {
    throw std::runtime_error(_impl->status);
}

void CudaBICGBackend::synchronize(double*) {
    throw std::runtime_error(_impl->status);
}

// ---------------------------------------------------------------------------
// Batch arena -- same story: the class exists so the call sites compile, and
// every entry point refuses.  rasberySetBatchWidth() is the exception: main()
// calls it before it can know whether CUDA is there, and the refusal is raised
// where the arena is actually needed.
// ---------------------------------------------------------------------------

class CudaBatchArena::Impl {
public:
    std::string status = "CUDA support was not compiled";
};

CudaBatchArena::CudaBatchArena(Geometry&, int) : _impl(std::make_unique<Impl>()) {}
CudaBatchArena::~CudaBatchArena() = default;

bool CudaBatchArena::available() const { return false; }
const std::string& CudaBatchArena::status() const { return _impl->status; }
int CudaBatchArena::slots() const { return 0; }
BackendCounters CudaBatchArena::counters() const { return {}; }
bool CudaBatchArena::compatible(Geometry&) const { return false; }
int CudaBatchArena::acquireSlot() { return -1; }
void CudaBatchArena::releaseSlot(int) {}
void CudaBatchArena::stage(int, const double*, const double*, const double*, const double*) {
    throw std::runtime_error(_impl->status);
}
void CudaBatchArena::setInner(int, int, double) {
    throw std::runtime_error(_impl->status);
}
void CudaBatchArena::solve(int, double*) {
    throw std::runtime_error(_impl->status);
}
// The lease bookkeeping still runs in a stub build; no hook is installed, so
// no device call is made.  That keeps the owner destructors' unpin path on the
// SAME code in the MSVC/no-CUDA regression as in the CUDA build.
bool CudaBatchArena::pinHost(const void* p, size_t bytes, const char* tag) const {
    return rasberyPinHost(p, bytes, tag);
}
/// Rev.7.1 Task 9 link 2.  No device, so no device addresses: every pointer is
/// null and `valid` is false, which is exactly what the segment's eligibility
/// predicate tests before it binds anything.
CudaBatchArena::CmfdResidentView CudaBatchArena::residentView(int) const {
    return CmfdResidentView{};
}

void CudaBatchArena::stageSweeps(int, const CmfdSweepIO&) {
    throw std::runtime_error(_impl->status);
}
void CudaBatchArena::solveSweeps(int, double*, CmfdSweepIO&) {
    throw std::runtime_error(_impl->status);
}
void CudaBatchArena::solveCommon(int, double*, int) {
    throw std::runtime_error(_impl->status);
}
void CudaBatchArena::reportBatchOccupancy(const char*) const {}

namespace {
int g_batch_width = 0;
}

void rasberySetBatchWidth(int slots) { g_batch_width = slots > 0 ? slots : 0; }
int  rasberyBatchWidth() { return g_batch_width; }

/// Rev.7.1 Task 6 declared this in CudaBICGBackend.h and defined it only in the
/// CUDA arm, so BICGSolver's constructor -- which calls it unconditionally --
/// left the CPU-only link with an undefined reference.  There is no resident
/// single-instance CMFD without a device, so the answer here is false.
bool rasberyResidentSingleCmfd() { return false; }

CudaBatchArena* rasberyBatchArena(Geometry&) {
    throw std::runtime_error(
        "RASBERY batch mode requires a CUDA build (-DRASBERY_ENABLE_CUDA=ON)");
}

void rasberyReleaseBatchArena() {}

} // namespace rasbery
