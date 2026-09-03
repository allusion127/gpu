// GPU thermal hydraulics: XSSet::UpdateTH + XSSet::SolveTH on the device.
// Contract, cost ledger and gate class: CudaThBackend.h.
//
// The arithmetic below is not restated here at all -- it is src/ThKernel.h,
// which the host mining harness compiles from the same text.  That is the
// difference from CudaCramBackend.cu, which had to transcribe milk.h because
// milk.h cannot be handed to nvcc: everything T/H reads is plain arithmetic on
// doubles, so ONE body serves both compilers and there is nothing to drift.
// --fmad=false is set on this file in CMakeLists.txt for the reason every other
// bit-exact TU has it: nvcc must not contract a*b+c where the mined mask said
// the host compiler did not.

#include "CudaThBackend.h"

#include "GpuCaptureArbiter.h"
#include "ThFormMask.h"
#include "ThGpuReceipt.h"
#include "XferLedger.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace rasbery {

namespace {

namespace th = rasbery::th;

/// The project's usual truthiness, duplicated here rather than shared because
/// this TU sees neither CudaBICGBackend.cu's copy nor CudaXsReconBackend.cu's.
bool truthy(const char* v) {
    if (v == nullptr || *v == '\0') return false;
    const std::string s(v);
    return !(s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE");
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
//
// FOUR LAUNCHES, AND THE SPLIT IS THE HOST'S DEPENDENCY STRUCTURE, NOT A TUNING
// CHOICE.  Phase 1 is embarrassingly parallel over nodes; phase 2 is two folds
// whose ORDER is part of the answer and therefore one lane; phase 3 is parallel
// over channels and serial within one; phase 4 is parallel over nodes again with
// a max reduction.  Fusing 1 and 2 would need a grid-wide barrier, which is a
// cooperative launch, which is a portability constraint this arm does not need
// to take on for four microseconds of launch overhead.

__global__ void kernelNodePower(th::ThView v, unsigned long long forms) {
    const int lk = blockIdx.x * blockDim.x + threadIdx.x;
    if (lk >= v.nxyz) return;
    v.node_power[lk] = th::thNodePower(v, lk, forms);
}

/// The two order-dependent folds and everything derived from them, in ONE lane.
///
/// WHY ONE LANE AND NOT A REDUCTION.  `total_power += node_power[lk]` over
/// ascending lk is not associative in floating point; a block reduction returns
/// a different double, `norm = actual_power / total_raw_power` inherits the
/// difference, and norm multiplies EVERY node power in the sweep.  So the
/// difference would not stay in the last bits of one scalar, it would be applied
/// to the whole core.  20k serial adds is a few microseconds against a phase
/// that was 0.70 s on the host.
///
/// `scalars` is [0] = norm, [1] = flow_per_channel, [2] = total_raw_power.
__global__ void kernelFolds(th::ThView v, double* scalars) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // UpdateTH's sign flip, which SolveTH's `total_raw_power` then re-folds over
    // the flipped array.  The host does exactly this -- one fold, a conditional
    // negation of the array, then a SECOND fold inside SolveTH -- and the second
    // fold of a negated array is not the negation of the first fold, so both
    // must happen here too.
    const double total_power = th::thTotalPowerSerial(v.node_power, v.nxyz);
    if (total_power < 0.0)
        for (int lk = 0; lk < v.nxyz; ++lk) v.node_power[lk] = -v.node_power[lk];

    const double total_raw_power = th::thTotalPowerSerial(v.node_power, v.nxyz);
    const double norm = (total_raw_power > 0.0) ? v.actual_power / total_raw_power : 0.0;

    const int    k_mid      = th::thKMid(v.kbc, v.kec);
    const double total_area = th::thTotalAreaSerial(v, v.node_power, k_mid);

    const double flow_per_channel = (v.use_input_mass_flux != 0)
                                        ? v.input_mass_flux * 1.0e-4
                                        : v.total_flow / total_area;

    scalars[0] = norm;
    scalars[1] = flow_per_channel;
    scalars[2] = total_raw_power;
}

/// LANE PER CHANNEL.  `over` is [3 * nxy]: count, worst, node, so the reducer
/// can replay the host's first-wins tie-break in ascending channel order rather
/// than racing for it.
__global__ void kernelChannelSweep(th::ThView v, const double* scalars, double* over,
                                   unsigned long long forms) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= v.nxy) return;
    const th::ThChannelOverflow o = th::thChannelSweep(v, l, scalars[0], scalars[1], forms);
    over[3 * l + 0] = static_cast<double>(o.count);
    over[3 * l + 1] = o.worst;
    over[3 * l + 2] = static_cast<double>(o.node);
}

/// The relaxation blend and the delta_Dop max.  `out` is [2]: delta_dop and the
/// summed off-table node count.
///
/// TWO PASSES IN ONE KERNEL AND A SINGLE BLOCK, because delta_Dop must be read
/// off the RELAXED temperatures -- the host computes it after the blend -- and a
/// second launch to do that would be a second full pass over three arrays for a
/// scalar.  MAX is associative and exact, so the reduction is allowed to be a
/// tree; the block-stride loop below is the tree.
__global__ void kernelRelaxAndDelta(th::ThView v, const double* over, double* out,
                                    unsigned long long forms) {
    extern __shared__ double s_max[];
    const double             w = v.th_relaxation;

    double local = 0.0;
    for (int lk = blockIdx.x * blockDim.x + threadIdx.x; lk < v.nxyz;
         lk += blockDim.x * gridDim.x) {
        double t = v.tful[lk];
        if (w < 1.0) {
            t          = th::thRelaxNode(v.tful_old[lk], t, w, forms);
            v.tful[lk] = t;
            v.tmod[lk] = th::thRelaxNode(v.tmod_old[lk], v.tmod[lk], w, forms);
            v.dmod[lk] = th::thRelaxNode(v.dmod_old[lk], v.dmod[lk], w, forms);
        }
        const double d = th::thDeltaDopNode(t, v.tful_old[lk]);
        if (d > local) local = d;
    }

    s_max[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && s_max[threadIdx.x + stride] > s_max[threadIdx.x])
            s_max[threadIdx.x] = s_max[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x != 0) return;

    // ONE BLOCK OWNS THE SCALARS.  With gridDim.x == 1 this is the whole answer
    // and needs no atomic; the launch below fixes the grid at one block for
    // exactly that reason.  The off-table scan runs here too, in ascending
    // channel order, so the host's first-wins `worst` tie-break is reproduced
    // rather than raced for.
    if (blockIdx.x != 0) return;
    out[0] = s_max[0];
    double n_over = 0.0, worst = 0.0, worst_node = -1.0;
    for (int l = 0; l < v.nxy; ++l) {
        n_over += over[3 * l + 0];
        if (over[3 * l + 0] > 0.0 && over[3 * l + 1] > worst) {
            worst      = over[3 * l + 1];
            worst_node = over[3 * l + 2];
        }
    }
    out[1] = n_over;
    out[2] = worst;
    out[3] = worst_node;
}

} // namespace

// ---------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------

struct ThBackend::Impl {
    bool        enabled = false;
    bool        failed  = false;
    std::string status_text;
    int         device = -1;
    int         block  = 128;

    cudaStream_t stream = nullptr;

    // Shape this instance is sized for; a change re-allocates.
    int nxyz = 0, nxy = 0, nz = 0, ng = 0;

    // node-invariant, generation-held
    double* d_vol     = nullptr;
    double* d_hmesh_x = nullptr;
    double* d_hmesh_y = nullptr;
    double* d_hz      = nullptr;
    unsigned long long geom_generation  = 0;
    bool               geom_uploaded    = false;

    double* d_mod_t_x = nullptr; double* d_mod_t_y = nullptr; double* d_mod_t_v = nullptr;
    double* d_rho_x   = nullptr; double* d_rho_y   = nullptr; double* d_rho_v   = nullptr;
    double* d_tf_x    = nullptr; double* d_tf_y    = nullptr; double* d_tf_v    = nullptr;
    int     mod_t_nx = 0, mod_t_ny = 0, rho_nx = 0, rho_ny = 0, tf_nx = 0, tf_ny = 0;
    unsigned long long table_generation = 0;
    bool               tables_uploaded  = false;

    // per-update
    double* d_xskf       = nullptr;
    double* d_phif       = nullptr;
    int*    d_burn       = nullptr;
    double* d_node_power = nullptr;
    double* d_tful       = nullptr;
    double* d_tmod       = nullptr;
    double* d_dmod       = nullptr;
    double* d_tful_old   = nullptr;
    double* d_tmod_old   = nullptr;
    double* d_dmod_old   = nullptr;
    double* d_over       = nullptr; ///< [3 * nxy]
    double* d_scalars    = nullptr; ///< [3]
    double* d_out        = nullptr; ///< [4]
    double* h_out        = nullptr; ///< pinned, 4

    unsigned long long n_updates     = 0;
    unsigned long long n_bytes_h2d   = 0;
    unsigned long long n_bytes_d2h   = 0;
    unsigned long long n_bytes_elide = 0;
    unsigned long long forms         = 0;
    double             wall_ms       = 0.0;

    ~Impl() { release(); }

    void release() {
        // WP19.  Same rule, same reason as CudaCramBackend.cu: a deck's teardown
        // frees device memory and destroys a stream on ITS OWN thread while
        // sibling lanes are running, one of which may have a graph capture open.
        rasbery::AllocWindow _alloc_window("th.release");
        auto                 f = [](void* p) { if (p) cudaFree(p); };
        f(d_vol); f(d_hmesh_x); f(d_hmesh_y); f(d_hz);
        f(d_mod_t_x); f(d_mod_t_y); f(d_mod_t_v);
        f(d_rho_x); f(d_rho_y); f(d_rho_v);
        f(d_tf_x); f(d_tf_y); f(d_tf_v);
        f(d_xskf); f(d_phif); f(d_burn); f(d_node_power);
        f(d_tful); f(d_tmod); f(d_dmod);
        f(d_tful_old); f(d_tmod_old); f(d_dmod_old);
        f(d_over); f(d_scalars); f(d_out);
        if (h_out) cudaFreeHost(h_out);
        if (stream) cudaStreamDestroy(stream);
        d_vol = d_hmesh_x = d_hmesh_y = d_hz = nullptr;
        d_mod_t_x = d_mod_t_y = d_mod_t_v = nullptr;
        d_rho_x = d_rho_y = d_rho_v = nullptr;
        d_tf_x = d_tf_y = d_tf_v = nullptr;
        d_xskf = d_phif = nullptr; d_burn = nullptr; d_node_power = nullptr;
        d_tful = d_tmod = d_dmod = nullptr;
        d_tful_old = d_tmod_old = d_dmod_old = nullptr;
        d_over = d_scalars = d_out = nullptr; h_out = nullptr;
        stream          = nullptr;
        geom_uploaded   = false;
        tables_uploaded = false;
    }

    bool fail(const char* what, cudaError_t rc) {
        failed      = true;
        status_text = std::string("disabled after CUDA failure in ") + what + ": " +
                      cudaGetErrorString(rc);
        std::fprintf(stderr,
                     "[RASBERY][TH_GPU][WARN] %s -- falling back to host SolveTH\n",
                     status_text.c_str());
        release();
        return false;
    }

    /// Refuse rather than run: a deck this kernel does not serve.  A decline is
    /// not a failure -- the arm stays available for the next call.
    bool decline(const char* why) {
        status_text = std::string("declined: ") + why;
        return false;
    }

    bool ensureShape(int nxy_in, int nz_in, int ng_in) {
        const int nxyz_in = nxy_in * nz_in;
        if (nxyz == nxyz_in && nxy == nxy_in && nz == nz_in && ng == ng_in &&
            stream != nullptr)
            return true;
        release();
        nxy  = nxy_in;
        nz   = nz_in;
        nxyz = nxyz_in;
        ng   = ng_in;

        // The ordinal the RECEIPT reports is the one these buffers land on, and
        // that is decided here, not in the constructor: --batch-mode selects a
        // slot's device on the worker thread, after the XSSet already exists.
        cudaGetDevice(&device);

        // WP19: every allocation and the stream creation are "potentially
        // unsafe" during a sibling lane's capture.
        rasbery::AllocWindow _alloc_window("th.shape.standup");
        // WP19.2: cudaStreamNonBlocking.  A LEGACY-BLOCKING stream is joined by
        // -- and joins -- the NULL stream process-wide, which is precisely the
        // coupling that invalidates a sibling lane's in-flight capture.
        cudaError_t rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (rc != cudaSuccess) return fail("cudaStreamCreateWithFlags", rc);

        const size_t n = static_cast<size_t>(nxyz);
        struct Alloc { void** p; size_t bytes; const char* name; };
        const std::vector<Alloc> allocs = {
            {(void**)&d_xskf,       n * ng * sizeof(double), "xskf"},
            {(void**)&d_phif,       n * ng * sizeof(double), "phif"},
            {(void**)&d_burn,       n * sizeof(int),         "burn"},
            {(void**)&d_node_power, n * sizeof(double),      "node_power"},
            {(void**)&d_tful,       n * sizeof(double),      "tful"},
            {(void**)&d_tmod,       n * sizeof(double),      "tmod"},
            {(void**)&d_dmod,       n * sizeof(double),      "dmod"},
            {(void**)&d_tful_old,   n * sizeof(double),      "tful_old"},
            {(void**)&d_tmod_old,   n * sizeof(double),      "tmod_old"},
            {(void**)&d_dmod_old,   n * sizeof(double),      "dmod_old"},
            {(void**)&d_vol,        n * sizeof(double),      "vol"},
            {(void**)&d_hmesh_x,    n * sizeof(double),      "hmesh_x"},
            {(void**)&d_hmesh_y,    n * sizeof(double),      "hmesh_y"},
            {(void**)&d_hz,         static_cast<size_t>(nz) * sizeof(double), "hz"},
            {(void**)&d_over,       static_cast<size_t>(3 * nxy) * sizeof(double), "over"},
            {(void**)&d_scalars,    3 * sizeof(double),      "scalars"},
            {(void**)&d_out,        4 * sizeof(double),      "out"},
        };
        for (const Alloc& a : allocs) {
            rc = cudaMalloc(a.p, a.bytes);
            if (rc != cudaSuccess) return fail(a.name, rc);
        }
        rc = cudaMallocHost((void**)&h_out, 4 * sizeof(double));
        if (rc != cudaSuccess) return fail("cudaMallocHost(out)", rc);
        return true;
    }

    bool uploadTables(const thgpu::TableView& t) {
        if (tables_uploaded && table_generation == t.generation) return true;
        rasbery::AllocWindow _alloc_window("th.tables");
        auto                 f = [](void* p) { if (p) cudaFree(p); };
        f(d_mod_t_x); f(d_mod_t_y); f(d_mod_t_v);
        f(d_rho_x); f(d_rho_y); f(d_rho_v);
        f(d_tf_x); f(d_tf_y); f(d_tf_v);
        d_mod_t_x = d_mod_t_y = d_mod_t_v = nullptr;
        d_rho_x = d_rho_y = d_rho_v = nullptr;
        d_tf_x = d_tf_y = d_tf_v = nullptr;

        struct Axis { double** d; const double* h; int n; const char* name; };
        const Axis axes[] = {
            {&d_mod_t_x, t.mod_t_x, t.mod_t_nx, "mod_t.x"},
            {&d_mod_t_y, t.mod_t_y, t.mod_t_ny, "mod_t.y"},
            {&d_mod_t_v, t.mod_t_v, t.mod_t_nx * t.mod_t_ny, "mod_t.v"},
            {&d_rho_x, t.mod_rho_x, t.mod_rho_nx, "mod_rho.x"},
            {&d_rho_y, t.mod_rho_y, t.mod_rho_ny, "mod_rho.y"},
            {&d_rho_v, t.mod_rho_v, t.mod_rho_nx * t.mod_rho_ny, "mod_rho.v"},
            {&d_tf_x, t.tf_x, t.tf_nx, "tf.x"},
            {&d_tf_y, t.tf_y, t.tf_ny, "tf.y"},
            {&d_tf_v, t.tf_v, t.tf_nx * t.tf_ny, "tf.v"},
        };
        for (const Axis& a : axes) {
            const size_t bytes = static_cast<size_t>(a.n) * sizeof(double);
            cudaError_t  rc    = cudaMalloc((void**)a.d, bytes);
            if (rc != cudaSuccess) return fail(a.name, rc);
            rc = rasbery::xfer::memcpyAsync("CudaThBackend.cu:tables", a.name, *a.d, a.h,
                                            bytes, cudaMemcpyHostToDevice, stream);
            if (rc != cudaSuccess) return fail(a.name, rc);
            n_bytes_h2d += bytes;
        }
        mod_t_nx = t.mod_t_nx; mod_t_ny = t.mod_t_ny;
        rho_nx   = t.mod_rho_nx; rho_ny = t.mod_rho_ny;
        tf_nx    = t.tf_nx; tf_ny = t.tf_ny;
        table_generation = t.generation;
        tables_uploaded  = true;
        return true;
    }

    bool uploadGeom(const thgpu::GeomView& g) {
        if (geom_uploaded && geom_generation == g.generation) return true;
        struct Col { double* d; const double* h; size_t n; const char* name; };
        const Col cols[] = {
            {d_vol, g.vol, static_cast<size_t>(nxyz), "vol"},
            {d_hmesh_x, g.hmesh_x, static_cast<size_t>(nxyz), "hmesh_x"},
            {d_hmesh_y, g.hmesh_y, static_cast<size_t>(nxyz), "hmesh_y"},
            {d_hz, g.hz, static_cast<size_t>(nz), "hz"},
        };
        for (const Col& c : cols) {
            const size_t      bytes = c.n * sizeof(double);
            const cudaError_t rc =
                rasbery::xfer::memcpyAsync("CudaThBackend.cu:geom", c.name, c.d, c.h, bytes,
                                           cudaMemcpyHostToDevice, stream);
            if (rc != cudaSuccess) return fail(c.name, rc);
            n_bytes_h2d += bytes;
        }
        geom_generation = g.generation;
        geom_uploaded   = true;
        return true;
    }
};

ThBackend::ThBackend() : _impl(new Impl) {
    _impl->enabled = truthy(std::getenv("RASBERY_GPU_TH"));
    if (!_impl->enabled) {
        _impl->status_text = "off (RASBERY_GPU_TH unset)";
        return;
    }
    int         count = 0;
    cudaError_t rc    = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess || count <= 0) {
        _impl->enabled     = false;
        _impl->status_text = "no CUDA device";
        return;
    }
    cudaGetDevice(&_impl->device);
    if (const char* b = std::getenv("RASBERY_GPU_TH_BLOCK")) {
        const int v = std::atoi(b);
        if (v == 32 || v == 64 || v == 128 || v == 256) _impl->block = v;
    }
    _impl->status_text = "on";
}

ThBackend::~ThBackend() = default;

bool ThBackend::available() const { return _impl->enabled && !_impl->failed; }

const std::string& ThBackend::status() const { return _impl->status_text; }

unsigned long long ThBackend::deviceUpdates() const { return _impl->n_updates; }
unsigned long long ThBackend::bytesElided() const { return _impl->n_bytes_elide; }
unsigned long long ThBackend::bytesH2d() const { return _impl->n_bytes_h2d; }
unsigned long long ThBackend::bytesD2h() const { return _impl->n_bytes_d2h; }
double             ThBackend::wallMs() const { return _impl->wall_ms; }
int                ThBackend::deviceOrdinal() const { return _impl->device; }
unsigned long long ThBackend::formsMask() const { return _impl->forms; }

bool ThBackend::solveTh(const thgpu::TableView& tables, const thgpu::GeomView& geom,
                        const thgpu::UpdateView& v, double& delta_dop) {
    Impl& s = *_impl;
    if (!s.enabled || s.failed) return false;

    // Shapes the lane-per-channel mapping cannot serve.  DECLINE, do not clamp:
    // a kernel that quietly ran a fraction of the core would produce a plausible
    // temperature field and a wrong one.
    if (v.nxy <= 0 || v.nz <= 0 || v.ng <= 0) return s.decline("degenerate shape");
    if (v.nxy * v.nz != v.nxyz) return s.decline("nxy * nz != nxyz");
    if (v.kbc < 0 || v.kec > v.nz || v.kbc > v.kec) return s.decline("kbc/kec out of range");
    if (v.tful == nullptr || v.tmod == nullptr || v.dmod == nullptr)
        return s.decline("no host output arrays");

    const auto t0 = std::chrono::steady_clock::now();

    if (!s.ensureShape(v.nxy, v.nz, v.ng)) return false;
    if (!s.uploadTables(tables)) return false;
    if (!s.uploadGeom(geom)) return false;

    const size_t n     = static_cast<size_t>(v.nxyz);
    const size_t nb    = n * sizeof(double);
    const size_t nng_b = n * static_cast<size_t>(v.ng) * sizeof(double);
    cudaError_t  rc    = cudaSuccess;

    // THE BORROW.  When the caller handed us device addresses for the flux and
    // the live macroscopic block, the two biggest uploads of this call do not
    // happen at all -- these are bytes the device already holds, written by the
    // solve that produced them.  The event is what orders this stream behind
    // that solve; without it the borrow would be a race, not an elision.
    const bool borrow = v.xskf_device != nullptr && v.phif_device != nullptr;
    if (borrow) {
        if (v.device_ready != nullptr) {
            rc = cudaStreamWaitEvent(s.stream, static_cast<cudaEvent_t>(v.device_ready), 0);
            if (rc != cudaSuccess) return s.fail("cudaStreamWaitEvent", rc);
        }
        s.n_bytes_elide += 2 * nng_b;
        rasbery::xfer::countElisionTest(true, 2 * nng_b);
    } else {
        if (v.xskf == nullptr || v.phif == nullptr) return s.decline("no flux/xskf source");
        rasbery::xfer::countElisionTest(false, 2 * nng_b);
        rc = rasbery::xfer::memcpyAsync("CudaThBackend.cu:solveTh", "xskf", s.d_xskf, v.xskf,
                                        nng_b, cudaMemcpyHostToDevice, s.stream);
        if (rc != cudaSuccess) return s.fail("H2D xskf", rc);
        rc = rasbery::xfer::memcpyAsync("CudaThBackend.cu:solveTh", "phif", s.d_phif, v.phif,
                                        nng_b, cudaMemcpyHostToDevice, s.stream);
        if (rc != cudaSuccess) return s.fail("H2D phif", rc);
        s.n_bytes_h2d += 2 * nng_b;
    }

    if (v.burn == nullptr) return s.decline("no burnup key");
    rc = rasbery::xfer::memcpyAsync("CudaThBackend.cu:solveTh", "burn", s.d_burn, v.burn,
                                    n * sizeof(int), cudaMemcpyHostToDevice, s.stream);
    if (rc != cudaSuccess) return s.fail("H2D burn", rc);
    s.n_bytes_h2d += n * sizeof(int);

    // The three temperature arrays go up TWICE: once as the working copy the
    // sweep overwrites, once as the SNAPSHOT the relaxation blends against.  The
    // host makes the same two copies (`tful_old` and the live array); doing it
    // D2D on the device would need a fourth launch to no purpose, since the
    // upload is happening anyway.
    struct Pair { double* dst; double* snap; const double* src; const char* name; };
    const Pair pairs[] = {
        {s.d_tful, s.d_tful_old, v.tful, "tful"},
        {s.d_tmod, s.d_tmod_old, v.tmod, "tmod"},
        {s.d_dmod, s.d_dmod_old, v.dmod, "dmod"},
    };
    for (const Pair& p : pairs) {
        rc = rasbery::xfer::memcpyAsync("CudaThBackend.cu:solveTh", p.name, p.dst, p.src, nb,
                                        cudaMemcpyHostToDevice, s.stream);
        if (rc != cudaSuccess) return s.fail("H2D state", rc);
        rc = rasbery::xfer::memcpyAsync("CudaThBackend.cu:solveTh", "snapshot", p.snap, p.dst,
                                        nb, cudaMemcpyDeviceToDevice, s.stream);
        if (rc != cudaSuccess) return s.fail("D2D snapshot", rc);
        s.n_bytes_h2d += nb;
    }

    th::ThView k{};
    k.nxy                  = v.nxy;
    k.nz                   = v.nz;
    k.nxyz                 = v.nxyz;
    k.ng                   = v.ng;
    k.kbc                  = v.kbc;
    k.kec                  = v.kec;
    k.pressure             = v.pressure;
    k.inlet_h              = v.inlet_h;
    k.actual_power         = v.actual_power;
    k.total_flow           = v.total_flow;
    k.input_mass_flux      = v.input_mass_flux;
    k.use_input_mass_flux  = v.use_input_mass_flux;
    k.fuel_temp_rise_scale = v.fuel_temp_rise_scale;
    k.fuel_rods_per_node   = v.fuel_rods_per_node;
    k.th_relaxation        = v.th_relaxation;
    k.h_table_max          = v.h_table_max;
    k.xskf = borrow ? static_cast<const double*>(v.xskf_device) : s.d_xskf;
    k.phif = borrow ? static_cast<const double*>(v.phif_device) : s.d_phif;
    k.vol     = s.d_vol;
    k.hmesh_x = s.d_hmesh_x;
    k.hmesh_y = s.d_hmesh_y;
    k.hz      = s.d_hz;
    k.burn    = s.d_burn;
    k.mod_t   = {s.d_mod_t_x, s.d_mod_t_y, s.d_mod_t_v, s.mod_t_nx, s.mod_t_ny};
    k.mod_rho = {s.d_rho_x, s.d_rho_y, s.d_rho_v, s.rho_nx, s.rho_ny};
    k.tf      = {s.d_tf_x, s.d_tf_y, s.d_tf_v, s.tf_nx, s.tf_ny};
    k.node_power = s.d_node_power;
    k.tful_old   = s.d_tful_old;
    k.tmod_old   = s.d_tmod_old;
    k.dmod_old   = s.d_dmod_old;
    k.tful       = s.d_tful;
    k.tmod       = s.d_tmod;
    k.dmod       = s.d_dmod;

    // THE MASK IS RESOLVED HERE AND PASSED AS AN ARGUMENT.  thFormMask() mines
    // on first call, which is why this line is inside the armed path: a run that
    // never reaches here never mines and never prints a [RASBERY][FORMS] line
    // about a feature it did not use.
    const unsigned long long forms = th::thFormMask();
    s.forms                        = forms;

    const int grid_nodes = (v.nxyz + s.block - 1) / s.block;
    const int grid_chan  = (v.nxy + s.block - 1) / s.block;

    kernelNodePower<<<grid_nodes, s.block, 0, s.stream>>>(k, forms);
    kernelFolds<<<1, 1, 0, s.stream>>>(k, s.d_scalars);
    kernelChannelSweep<<<grid_chan, s.block, 0, s.stream>>>(k, s.d_scalars, s.d_over, forms);
    // ONE BLOCK: kernelRelaxAndDelta owns the scalar outputs and reduces without
    // an atomic, which is only sound while gridDim.x == 1.  The block-stride
    // loop inside it is what covers nxyz.
    kernelRelaxAndDelta<<<1, s.block, static_cast<size_t>(s.block) * sizeof(double),
                          s.stream>>>(k, s.d_over, s.d_out, forms);

    rc = cudaGetLastError();
    if (rc != cudaSuccess) return s.fail("kernel launch", rc);

    // THE SCALARS COME BACK BEFORE THE ARRAYS, and that ordering is the same
    // rule CudaCramBackend.cu's status-before-D2H is: delta_dop and the
    // off-table census decide whether the host may believe the arrays, so a
    // publish that raced them would publish a field its own diagnostic had not
    // cleared.
    rc = rasbery::xfer::memcpyAsync("CudaThBackend.cu:solveTh", "scalars", s.h_out, s.d_out,
                                    4 * sizeof(double), cudaMemcpyDeviceToHost, s.stream);
    if (rc != cudaSuccess) return s.fail("D2H scalars", rc);
    if ((rc = rasbery::xfer::streamSync("CudaThBackend.cu:solveTh", "scalar drain",
                                        s.stream)) != cudaSuccess)
        return s.fail("scalar drain", rc);
    s.n_bytes_d2h += 4 * sizeof(double);

    if (!(s.h_out[0] >= 0.0)) return s.decline("non-finite delta_dop");

    // The three arrays.  NOT ELIDED, and CudaThBackend.h says why: their reader
    // (XSSet::BuildFlatXsStream) is still on the host.
    for (const Pair& p : pairs) {
        rc = rasbery::xfer::memcpyAsync("CudaThBackend.cu:solveTh", p.name,
                                        const_cast<double*>(p.src), p.dst, nb,
                                        cudaMemcpyDeviceToHost, s.stream);
        if (rc != cudaSuccess) return s.fail("D2H state", rc);
        s.n_bytes_d2h += nb;
    }
    if ((rc = rasbery::xfer::streamSync("CudaThBackend.cu:solveTh", "final drain",
                                        s.stream)) != cudaSuccess)
        return s.fail("final drain", rc);

    delta_dop = s.h_out[0];

    // The host's water-property-table warning, printed from the device census
    // rather than re-derived: same threshold, same worst node, same sentence.
    const int n_over = static_cast<int>(s.h_out[1]);
    if (n_over > 0) {
        std::fprintf(stderr,
                     "[RASBERY][WARN][th] %d of %d nodes ran off the water-property table "
                     "(enthalpy axis ends at %.1f kJ/kg); their coolant temperature and "
                     "density are CLAMPED at the table edge and no longer respond to power. "
                     "Worst %.1f kJ/kg at node %d (excess %.1f). The single-phase "
                     "closed-channel model is outside its range here -- check the radial "
                     "peaking against the core flow.\n",
                     n_over, v.nxyz, v.h_table_max, s.h_out[2], static_cast<int>(s.h_out[3]),
                     s.h_out[2] - v.h_table_max);
    }

    ++s.n_updates;
    s.wall_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
    return true;
}

} // namespace rasbery
