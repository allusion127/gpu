#include "BICGSolver.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define diag(igs, ige, l)   diag[(l) * _g.ng2() + (ige) * _g.ng() + (igs)]
#define cc(lr, idir, ig, l) cc[(l) * _g.ng() * NDIRMAX * LR + (ig) * NDIRMAX * LR + (idir) * LR + (lr)]
#define src(ig, l)          src[(l) * _g.ng() + (ig)]
#define aphi(ig, l)         aphi[(l) * _g.ng() + (ig)]
#define phi(ig, l)          phi[(l) * _g.ng() + (ig)]

#define vr(ig, l)  _vr[((l) * _g.ng()) + (ig)]
#define vr0(ig, l) _vr0[((l) * _g.ng()) + (ig)]
#define vp(ig, l)  _vp[((l) * _g.ng()) + (ig)]
#define vv(ig, l)  _vv[((l) * _g.ng()) + (ig)]
#define vs(ig, l)  _vs[((l) * _g.ng()) + (ig)]
#define vt(ig, l)  _vt[((l) * _g.ng()) + (ig)]
#define vy(ig, l)  _vy[((l) * _g.ng()) + (ig)]
#define vz(ig, l)  _vz[((l) * _g.ng()) + (ig)]

using namespace rasbery;

BICGSolver::BICGSolver(Geometry& g)
    : _g(g), _diag_ptr(nullptr), _cuda(nullptr), _use_cuda(false),
      _arena(nullptr), _batch_slot(-1) {
    const size_t nv = static_cast<size_t>(_g.ng()) * _g.nxyz();
    const size_t nm = static_cast<size_t>(_g.ng2()) * _g.nxyz();

    // BiCGSTAB workspace
    _vz.assign(nv, 0.0);
    _vy.assign(nv, 0.0);
    _vr.assign(nv, 0.0);
    _vr0.assign(nv, 0.0);
    _vp.assign(nv, 0.0);
    _vv.assign(nv, 0.0);
    _vs.assign(nv, 0.0);
    _vt.assign(nv, 0.0);

    // SSOR preconditioner
    _dinv.assign(nm, 0.0);
    _ssor_tmp.assign(nv, 0.0);

    // Batch mode wins over the per-instance backend: the arena *is* the CUDA
    // backend for every instance in the batch, and a private one would take a
    // second copy of the operator onto the device for nothing.
    //
    // RASBERY_BATCH_CPU=1 keeps the instance-parallel host orchestration but
    // solves CMFD on the CPU.  It exists as the control experiment for the
    // batch mode -- "is this the arena or is this the rest of RASBERY?" -- and
    // doubles as an in-process multi-instance runner on machines with no GPU.
    const char* gpu_env = std::getenv("RASBERY_GPU");
    const bool  gpu_requested = gpu_env != nullptr && std::string(gpu_env) != "0";

    // Rev.7.1 Task 6: RESIDENT SINGLE.  Before this, the arena existed only
    // under --batch-mode, so canUseDeviceAssembly() (BICGCMFD.cpp:207-217) --
    // which requires `_ls->arena() != nullptr` -- refused for every normal
    // single run.  The resident device assembly and the device-resident sweeps
    // were therefore reachable only by asking for a batch of one, which is not
    // how anybody runs a single deck.
    //
    // RASBERY_GPU_CMFD_RESIDENT_SINGLE opens the same arena at width 1.  It
    // still needs RASBERY_GPU: this is a CUDA path, and a flag that silently
    // turned the GPU on would be a worse surprise than one that does nothing.
    const bool resident_single =
        rasberyBatchWidth() == 0 && rasberyResidentSingleCmfd() && gpu_requested;

    if ((rasberyBatchWidth() > 0 || resident_single) &&
        std::getenv("RASBERY_BATCH_CPU") == nullptr) {
        _arena      = rasberyBatchArena(_g);
        _batch_slot = _arena->acquireSlot();
        if (_batch_slot < 0)
            throw std::runtime_error(
                resident_single
                    ? "RASBERY_GPU_CMFD_RESIDENT_SINGLE: the width-1 arena is already "
                      "taken; resident-single supports exactly one concurrent instance"
                    : "batch mode: no free instance slot (more concurrent instances than --batch-mode M)");
        _use_cuda = true;
        if (resident_single)
            std::cout << "[RASBERY][CUDA] CMFD resident-single arena (width 1)" << std::endl;
        return;
    }

    if (gpu_requested) {
        _cuda = std::make_unique<CudaBICGBackend>(_g);
        if (!_cuda->available())
            throw std::runtime_error("RASBERY_GPU requested but unavailable: " + _cuda->status());
        _use_cuda = true;
        std::cout << "[RASBERY][CUDA] CMFD BiCGSTAB backend: " << _cuda->status() << std::endl;
    }
}

BICGSolver::~BICGSolver() {
    if (_arena != nullptr) {
        // Counters are reported once for the whole arena at teardown, not once
        // per instance: they are shared device-side tallies.
        _arena->releaseSlot(_batch_slot);
        _batch_slot = -1;
        _arena      = nullptr;
        return;
    }
    if (!_cuda) return;
    const BackendCounters c = _cuda->counters();
    std::cout
        << "[RASBERY][CUDA][BACKEND_COUNTERS] {"
        << "\"xs_gpu_calls\":" << c.xs_gpu_calls << ','
        << "\"xs_cpu_fallbacks\":" << c.xs_cpu_fallbacks << ','
        << "\"cmfd_gpu_calls\":" << c.cmfd_gpu_calls << ','
        << "\"cmfd_cpu_fallbacks\":" << c.cmfd_cpu_fallbacks << ','
        << "\"cmfd_assembly_gpu_calls\":" << c.cmfd_assembly_gpu_calls << ','
        << "\"cmfd_assembly_cpu_fallbacks\":"
        << c.cmfd_assembly_cpu_fallbacks << ','
        << "\"cmfd_diag_h2d_elided_bytes\":"
        << c.cmfd_diag_h2d_elided_bytes << ','
        << "\"cmfd_cc_h2d_elided_bytes\":"
        << c.cmfd_cc_h2d_elided_bytes << ','
        << "\"cmfd_psi_h2d_elided_bytes\":"
        << c.cmfd_psi_h2d_elided_bytes << ','
        << "\"cmfd_psi_d2h_elided_bytes\":"
        << c.cmfd_psi_d2h_elided_bytes << ','
        << "\"cmfd_phi_mirror_ns\":" << c.cmfd_phi_mirror_ns << ','
        << "\"cmfd_phi_mirror_calls\":" << c.cmfd_phi_mirror_calls << ','
        << "\"cmfd_phi_mirror_bypassed\":" << c.cmfd_phi_mirror_bypassed << ','
        << "\"cmfd_phi_h2d_elided_bytes\":"
        << c.cmfd_phi_h2d_elided_bytes << ','
        << "\"bicg_early_convergence_exits\":"
        << c.bicg_early_convergence_exits << ','
        << "\"bicg_restarts\":" << c.bicg_restarts << ','
        << "\"nodal_gpu_calls\":" << c.nodal_gpu_calls << ','
        << "\"nodal_cpu_fallbacks\":" << c.nodal_cpu_fallbacks << ','
        << "\"th_gpu_calls\":" << c.th_gpu_calls << ','
        << "\"depletion_gpu_calls\":" << c.depletion_gpu_calls << ','
        << "\"bulk_h2d_calls_during_iteration\":"
        << c.bulk_h2d_calls_during_iteration << ','
        << "\"bulk_h2d_skipped_during_iteration\":"
        << c.bulk_h2d_skipped_during_iteration << ','
        << "\"bulk_h2d_bytes_during_iteration\":"
        << c.bulk_h2d_bytes_during_iteration << ','
        << "\"bulk_d2h_calls_during_iteration\":"
        << c.bulk_d2h_calls_during_iteration << ','
        << "\"bulk_d2h_bytes_during_iteration\":"
        << c.bulk_d2h_bytes_during_iteration << ','
        << "\"status_d2h_calls_during_iteration\":"
        << c.status_d2h_calls_during_iteration << ','
        << "\"stream_sync_calls_during_iteration\":"
        << c.stream_sync_calls_during_iteration << ','
        << "\"graph_launches\":" << c.graph_launches << ','
        << "\"graph_reinstantiations\":" << c.graph_reinstantiations << ','
        << "\"graph_fallbacks\":" << c.graph_fallbacks << ','
        << "\"iter_batch\":" << c.iter_batch << ','
        << "\"batched_graph_launches\":" << c.batched_graph_launches << ','
        << "\"overrun_iterations\":" << c.overrun_iterations << ','
        << "\"fp32_active\":" << c.fp32_active << ','
        << "\"fp32_fallbacks\":" << c.fp32_fallbacks
        << "}" << std::endl;
}

// ============================================================
// Reset: compute initial residual r = b - A*x
// ============================================================

double BICGSolver::reset(const int& ig, const int& l, double* diag, double* cc, double* phi, double* src) {
    double aphi = axb(ig, l, diag, cc, phi);
    vr(ig, l)   = src(ig, l) - aphi;
    vr0(ig, l)  = vr(ig, l);
    vp(ig, l)   = 0.0;
    vv(ig, l)   = 0.0;

    return vr(ig, l) * vr(ig, l);
}

void BICGSolver::reset(double* diag, double* cc, double* phi, double* src, double& r20) {
    if (_use_cuda) {
        // Uploads only. r20 stays on the device, where the inner loop now
        // tests against it; the host has no use for it and fetching it would
        // cost the very pipeline drain this path exists to remove.
        if (_arena != nullptr)
            _arena->stage(_batch_slot, diag, cc, phi, src);
        else
            _cuda->reset(diag, cc, phi, src);
        r20 = 0.0;
        return;
    }

    _calpha = 1;
    _crho   = 1;
    _comega = 1;

    r20 = 0;
    for (int l = 0; l < _g.nxyz(); ++l)
        for (int ig = 0; ig < _g.ng(); ++ig)
            r20 += reset(ig, l, diag, cc, phi, src);

    r20 = sqrt(r20);
}

// ============================================================
// SSOR Preconditioner: facilu computes D^{-1} blocks
// ============================================================

void BICGSolver::facilu(double* diag, double* cc) {
    _diag_ptr      = diag;
    if (_use_cuda) return;
    const int nxyz = _g.nxyz();

    // Precompute inverse of 2x2 diagonal blocks
    for (int l = 0; l < nxyz; ++l) {
        const double a00 = diag[l * 4 + 0];
        const double a10 = diag[l * 4 + 1];
        const double a01 = diag[l * 4 + 2];
        const double a11 = diag[l * 4 + 3];

        const double rdet = 1.0 / (a00 * a11 - a10 * a01);

        _dinv[l * 4 + 0] = rdet * a11;
        _dinv[l * 4 + 1] = -rdet * a10;
        _dinv[l * 4 + 2] = -rdet * a01;
        _dinv[l * 4 + 3] = rdet * a00;
    }
}

// ============================================================
// SSOR Preconditioner Application: x = M^{-1} * b
// Symmetric Gauss-Seidel (ω=1):
//   Forward:  (D + L) * tmp = b
//   Backward: (D + U) * x   = D * tmp
// ============================================================

void BICGSolver::minv(double* cc, double* b, double* x) {
    const int     nxyz = _g.nxyz();
    double*       tmp  = _ssor_tmp.data();
    const double* dinv = _dinv.data();
    const double* dptr = _diag_ptr;

    // Forward sweep: for l = 0..nxyz-1
    //   tmp(l) = D^{-1} * (b(l) - sum_{ln < l} cc(l,ln) * tmp(ln))
    for (int l = 0; l < nxyz; ++l) {
        double b0 = b[l * 2 + 0];
        double b1 = b[l * 2 + 1];

        for (int idir = 0; idir < NDIRMAX; ++idir) {
            for (int lr = 0; lr < LR; ++lr) {
                int ln = _g.neib(lr, idir, l);
                if (ln >= 0 && ln < l) {
                    b0 -= cc[l * 12 + 0 * 6 + idir * 2 + lr] * tmp[ln * 2 + 0];
                    b1 -= cc[l * 12 + 1 * 6 + idir * 2 + lr] * tmp[ln * 2 + 1];
                }
            }
        }

        tmp[l * 2 + 0] = dinv[l * 4 + 0] * b0 + dinv[l * 4 + 1] * b1;
        tmp[l * 2 + 1] = dinv[l * 4 + 2] * b0 + dinv[l * 4 + 3] * b1;
    }

    // Backward sweep: for l = nxyz-1..0
    //   x(l) = D^{-1} * (D*tmp(l) - sum_{ln > l} cc(l,ln) * x(ln))
    for (int l = nxyz - 1; l >= 0; --l) {
        // D * tmp
        double b0 = dptr[l * 4 + 0] * tmp[l * 2 + 0] + dptr[l * 4 + 1] * tmp[l * 2 + 1];
        double b1 = dptr[l * 4 + 2] * tmp[l * 2 + 0] + dptr[l * 4 + 3] * tmp[l * 2 + 1];

        for (int idir = 0; idir < NDIRMAX; ++idir) {
            for (int lr = 0; lr < LR; ++lr) {
                int ln = _g.neib(lr, idir, l);
                if (ln >= 0 && ln > l) {
                    b0 -= cc[l * 12 + 0 * 6 + idir * 2 + lr] * x[ln * 2 + 0];
                    b1 -= cc[l * 12 + 1 * 6 + idir * 2 + lr] * x[ln * 2 + 1];
                }
            }
        }

        x[l * 2 + 0] = dinv[l * 4 + 0] * b0 + dinv[l * 4 + 1] * b1;
        x[l * 2 + 1] = dinv[l * 4 + 2] * b0 + dinv[l * 4 + 3] * b1;
    }
}

// ============================================================
// BiCGSTAB Iteration
// ============================================================

void BICGSolver::solveInner(int nmax, double eps) {
    if (!_use_cuda)
        throw std::logic_error("BICGSolver::solveInner is the CUDA-resident inner loop");
    if (_arena != nullptr) {
        // In batch mode nothing is launched yet: the arena needs the flux
        // pointer too, and that only arrives at the observation boundary
        // (synchronizeCudaFlux), which is the very next statement in the
        // caller.  Recording the budget here keeps the call sequence -- and so
        // the CPU-side control flow -- identical to the single-instance path.
        _arena->setInner(_batch_slot, nmax, eps);
        return;
    }
    _cuda->solveInner(nmax, eps);
}

void BICGSolver::solve(double* diag, double* cc, double& r20, double* phi, double& r2) {
    int n = _g.nxyz() * _g.ng();

    // Preconditioned BiCGSTAB
    double crhod = _crho;
    _crho        = milk::dot(static_cast<size_t>(n), _vr0.data(), 1, _vr.data(), 1);
    _cbeta       = _crho * _calpha / (crhod * _comega);

    for (int i = 0; i < n; ++i) {
        _vp[i] = _vr[i] + _cbeta * (_vp[i] - _comega * _vv[i]);
    }

    minv(cc, _vp.data(), _vy.data());
    axb(diag, cc, _vy.data(), _vv.data());

    double r0v = milk::dot(static_cast<size_t>(n), _vr0.data(), 1, _vv.data(), 1);

    if (abs(r0v) < 1.E-10) {
        // BiCGSTAB breakdown: phi is not advanced this call. Publish the residual
        // that actually corresponds to the current phi, otherwise the caller keeps
        // testing whatever r2 happened to hold before (typically 0 on the first
        // inner iteration, which reads as "converged").
        r2 = sqrt(milk::dot(static_cast<size_t>(n), _vr.data(), 1, _vr.data(), 1));
        return;
    }

    _calpha = _crho / r0v;

    for (int i = 0; i < n; ++i) {
        _vs[i] = _vr[i] - _calpha * _vv[i];
    }

    minv(cc, _vs.data(), _vz.data());
    axb(diag, cc, _vz.data(), _vt.data());

    double pts = milk::dot(static_cast<size_t>(n), _vs.data(), 1, _vt.data(), 1);
    double ptt = milk::dot(static_cast<size_t>(n), _vt.data(), 1, _vt.data(), 1);

    _comega = 0.0;
    if (ptt != 0.0) {
        _comega = pts / ptt;
    }

    for (int i = 0; i < n; ++i) {
        phi[i] += _calpha * _vy[i] + _comega * _vz[i];
        _vr[i] = _vs[i] - _comega * _vt[i];
    }

    // True residual of the updated iterate: r = s - omega*t, which is exactly the
    // _vr just written above. The previous form used sqrt(ptt) = ||A*M^-1*s||,
    // which is not a residual at all -- it is the norm of a matrix-vector product
    // and can stay large while the iterate is already converged (or, with a
    // badly scaled operator, shrink while the residual does not). It was also
    // pre-divided by r20 even though the caller divides by r20 again, so the
    // effective test was ||t|| / ||r0||^2 -- tolerance that silently tracks the
    // problem scaling. r2 is now the plain absolute residual norm; the single
    // relative test lives in the caller.
    r2 = sqrt(milk::dot(static_cast<size_t>(n), _vr.data(), 1, _vr.data(), 1));
}

void BICGSolver::synchronizeCudaFlux(double* phi) {
    if (_arena != nullptr) {
        _arena->solve(_batch_slot, phi);
        return;
    }
    if (_use_cuda) _cuda->synchronize(phi);
}

bool BICGSolver::driveSweepsCuda(double* phi, CudaBatchArena::CmfdSweepIO& io) {
    if (_arena == nullptr || _batch_slot < 0 || !_arena->available()) return false;
    _arena->stageSweeps(_batch_slot, io);
    _arena->solveSweeps(_batch_slot, phi, io);
    return true;
}

bool BICGSolver::enqueueSweepsCuda(double* phi, const CudaBatchArena::CmfdSweepIO& io,
                                   const CudaBatchArena::CmfdSweepProbeSink& probe) {
    if (_arena == nullptr || _batch_slot < 0 || !_arena->available()) return false;
    return _arena->enqueueSweeps(_batch_slot, phi, io, probe);
}

bool BICGSolver::finishSweepsCuda(CudaBatchArena::CmfdSweepIO& io) {
    if (_arena == nullptr || _batch_slot < 0) return false;
    return _arena->finishSweeps(_batch_slot, io);
}

void* BICGSolver::sweepStream() const {
    if (_arena == nullptr) return nullptr;
    return _arena->sweepStream();
}

void BICGSolver::syncSweepStream() {
    if (_arena != nullptr) _arena->syncSweepStream();
}

void BICGSolver::axb(double* diag, double* cc, double* phi, double* aphi) {
    const int ng   = _g.ng();
    const int ng2  = _g.ng2();
    const int nxyz = _g.nxyz();
    const int ncc  = ng * NDIRMAX * LR;

    // Geometry topology never changes after Initialize: build the compressed per-node
    // neighbor lists once instead of re-resolving _g.neib on every matvec call.
    if (_neib_count.empty()) {
        _neib_count.assign(static_cast<size_t>(nxyz), 0);
        _neib_node.assign(static_cast<size_t>(nxyz) * NDIRMAX * LR, 0);
        _neib_slot.assign(static_cast<size_t>(nxyz) * NDIRMAX * LR, 0);
        for (int l = 0; l < nxyz; ++l) {
            int nn = 0;
            for (int idir = 0; idir < NDIRMAX; ++idir)
                for (int lr = 0; lr < LR; ++lr) {
                    const int neighbor = _g.neib(lr, idir, l);
                    if (neighbor != -1) {
                        _neib_node[static_cast<size_t>(l) * NDIRMAX * LR + nn] = neighbor;
                        _neib_slot[static_cast<size_t>(l) * NDIRMAX * LR + nn] = idir * LR + lr;
                        ++nn;
                    }
                }
            _neib_count[static_cast<size_t>(l)] = nn;
        }
    }

#pragma omp parallel for schedule(static) if (nxyz > rasbery_omp_gate)
    for (int l = 0; l < nxyz; ++l) {
        const int     nn     = _neib_count[static_cast<size_t>(l)];
        const int*    nln    = _neib_node.data() + static_cast<size_t>(l) * NDIRMAX * LR;
        const int*    nslot  = _neib_slot.data() + static_cast<size_t>(l) * NDIRMAX * LR;
        const double* diag_l = diag + static_cast<size_t>(l) * ng2;
        const double* cc_l   = cc + static_cast<size_t>(l) * ncc;
        const double* phi_l  = phi + static_cast<size_t>(l) * ng;
        for (int ig = 0; ig < ng; ++ig) {
            double ab = 0.0;
            for (int igs = 0; igs < ng; ++igs)
                ab += diag_l[ig * ng + igs] * phi_l[igs];
            for (int k = 0; k < nn; ++k)
                ab += cc_l[ig * NDIRMAX * LR + nslot[k]] * phi[static_cast<size_t>(nln[k]) * ng + ig];
            aphi[static_cast<size_t>(l) * ng + ig] = ab;
        }
    }
}

double BICGSolver::axb(const int& ig, const int& l, double* diag, double* cc, double* phi) {
    double ab = 0.0;
    for (int igs = 0; igs < _g.ng(); ++igs) {
        ab += diag(igs, ig, l) * phi(igs, l);
    }

    for (int idir = 0; idir < NDIRMAX; ++idir) {
        for (int lr = 0; lr < LR; ++lr) {
            int ln = _g.neib(lr, idir, l);
            if (ln != -1)
                ab += cc(lr, idir, ig, l) * phi(ig, ln);
        }
    }

    return ab;
}
