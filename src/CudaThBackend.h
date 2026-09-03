#pragma once

// The device thermal-hydraulics arm -- WP22, behind RASBERY_GPU_TH (default
// off).  The bodies are src/ThKernel.h, shared verbatim with the host mining
// harness; this header is the contract, the cost ledger and the gate class.
//
// ---------------------------------------------------------------------------
// WHAT THIS BACKEND OWNS
// ---------------------------------------------------------------------------
//
// The per-node NODE STATE of a statepoint: node_power, tful, tmod, dmod and the
// pre-update snapshots the under-relaxation blend needs, plus the three water
// property tables and the geometry columns SolveTH reads.  It is one object per
// XSSet, created on first use, exactly like _xsrecon_backend and _cram_backend.
//
// ---------------------------------------------------------------------------
// GATE CLASS: B0 IS THE TARGET, N1 IS THE CLAIM UNTIL A HOST MEASURES IT
// ---------------------------------------------------------------------------
//
// Every expression the kernels evaluate is +, -, * or / on doubles.  There is
// no transcendental, no libm call, no complex division -- which is exactly what
// makes this different from CudaCramBackend.h, whose CLASS N1 is forced by
// __divdc3.  So bit-identity is REACHABLE here, and three things are in place to
// reach it:
//
//   * every multiply-add reads its form from a mask MINED on this host against
//     a verbatim quotation in its own TU (src/ThFormMiner.cpp);
//   * the .cu is compiled --fmad=false so nvcc cannot fuse what the mask did
//     not ask for;
//   * the two order-dependent folds -- `total_power` over nodes and
//     `total_area` over channels -- run in ONE device lane in ascending index
//     order, because a tree reduction of a non-associative fold is a different
//     number and the norm it produces multiplies every node power.
//
// REACHABLE IS NOT MEASURED -- AND 238 HAS NOW MEASURED PART OF IT.  Block 48 of
// the 2026-08-30 pricing log ran the arm on kngr_238 and swept
// `RASBERY_TH_FORMS` over {0x00, 0x54, 0x57, 0x1f3}:
//
//   * with 0x57 the run reproduced the flag-off digest 1f36e75dc00ed2b4
//     EXACTLY -- h5diff rc=0, 0 lines.  THE ARITHMETIC OF THIS ARM IS B0 ON THIS
//     DECK.  There is no residual to explain, no libm difference, nothing left
//     over: under the right mask the device T/H is the host T/H, bit for bit.
//   * with the mask the binary MINED that day (0x54) it moved 866 lines, which
//     is what put the arm in N1 in the first place.
//
// So the open question is no longer "can the device reproduce the host" but
// "does the mining return the mask the host actually ran".  The gates are
// unchanged -- Gate A against the FP64 host trajectory, Gate B against MASTER --
// and the class quoted without a forced mask stays N1 until a 238 run shows
// `[RASBERY][FORMS] {"mask":"TH_FORMS",...,"value":"0x57","source":"mined"...}`
// and reproduces the digest with NO override in the environment.
// src/ThGpuReceipt.h carries the same sentence in the string a gate script reads.
//
// AND THE REASON THE MINING WAS NOT ENOUGH IS NOT THE ONE THIS HEADER EXPECTED.
// The suspicion was WP7-C's: a quotation is not a call site, the production
// spelling lives inside XSSet::SolveTH where milk::Table::Get and XSSet::GetTfuel
// are inlined into a much larger function, and gcc decides contraction THERE.
// That gap is real and measured (the same operands scored against the
// quotation's OUT-OF-LINE refGetTfuel and against its INLINED use pin bit
// TH_LERP_X0 differently, 17 mismatches versus 0 over a 20k sweep), and
// ThReference.cpp now quotes the call graph rather than only the expressions:
// one refSolveTH holding the channel loop, with tableGet / getTmod / getDmod /
// getTfuel inline and internal, as milk.h and XSSet.h spell them.
//
// BUT THAT WAS NOT WHAT PRODUCED 0x54, and the difference matters because the
// wrong diagnosis would have been "fixed" and shipped again.  The fixture never
// REACHED the two x-lerps: scoreMask derives node power from xskf/phif/vol
// through the shipped fold (nothing reads Fixture::node_power), buildFixture's
// `norm` was 1.0e-3 where SolveTH's is of order 1e2, so all 1,280 fuel nodes came
// out at `lpd < 0.03` W/cm and every tf query clamped to the first LPD knot with
// `fx == 0`.  0x54 and 0x57 both scored ZERO; the descent returned its seed and
// `mineStable` truthfully reported a zero residual.  With the operands repaired,
// BOTH the old shape and the new one mine 0x57.
//
// The check that was missing now exists: thmine::dontCareMask asks of every site
// whether the fixture can tell its forms apart, and src/ThFormMiner.cpp warns on
// anything but TH_HAVG.  src/XeFormAudit.h describes the same class of gap for
// the Anderson algebra.
//
// ---------------------------------------------------------------------------
// WHAT IS SAVED, AND WHAT IS NOT -- STATED SO IT CAN BE CHECKED
// ---------------------------------------------------------------------------
//
// SAVED: the 0.70 s of host T/H per statepoint, and the node_power array, which
// the host used to form and which now never exists on the host at all.  Under
// RASBERY_GPU_SHARED_STATE the kernel also BORROWS the canonical device flux and
// live macroscopic block instead of uploading phif and xskf, which is the round
// trip the plan review calls out; `bytes_elided` counts it.
//
// NOT SAVED, and the receipt says so in `bytes_d2h` rather than leaving it to be
// discovered: tful/tmod/dmod still come back to the host every update.  Their
// consumer is XSSet::BuildFlatXsStream, which resolves the flat-XS branch
// stream on the host and reads all three per node.  Moving THAT is WP13's
// elision, not this arm's; an arm that claimed the download away without moving
// its reader would be claiming a saving somebody else has to pay for.

#include "ThKernel.h"

#include <memory>
#include <string>

namespace rasbery {

/// Everything one device T/H update needs that is not already in ThView, plus
/// the host-side outputs it fills.  ThView's pointers are DEVICE addresses by
/// the time a kernel sees them; the pointers here are HOST addresses.
namespace thgpu {

/// The three water-property tables, as the host holds them.  Uploaded once and
/// re-uploaded only when `generation` changes -- they are read from CSV at load
/// and never move afterwards, so in practice that is exactly once.
struct TableView {
    unsigned long long generation = 0;

    const double* mod_t_x = nullptr;
    const double* mod_t_y = nullptr;
    const double* mod_t_v = nullptr;
    int           mod_t_nx = 0, mod_t_ny = 0;

    const double* mod_rho_x = nullptr;
    const double* mod_rho_y = nullptr;
    const double* mod_rho_v = nullptr;
    int           mod_rho_nx = 0, mod_rho_ny = 0;

    const double* tf_x = nullptr;
    const double* tf_y = nullptr;
    const double* tf_v = nullptr;
    int           tf_nx = 0, tf_ny = 0;
};

/// The geometry columns SolveTH reads.  Same residency rule as the tables: the
/// mesh does not move inside a run.
struct GeomView {
    unsigned long long generation = 0;

    const double* vol     = nullptr; ///< [nxyz]
    const double* hmesh_x = nullptr; ///< [nxyz]
    const double* hmesh_y = nullptr; ///< [nxyz]
    const double* hz      = nullptr; ///< [nz]
};

/// One T/H update.
struct UpdateView {
    int nxy  = 0;
    int nz   = 0;
    int nxyz = 0;
    int ng   = 0;
    int kbc  = 0;
    int kec  = 0;

    double pressure             = 0.0;
    double inlet_h              = 0.0;
    double actual_power         = 0.0;
    double total_flow           = 0.0;
    double input_mass_flux      = 0.0;
    int    use_input_mass_flux  = 0;
    double fuel_temp_rise_scale = 1.0;
    double fuel_rods_per_node   = 0.0; ///< Geometry::fuel_rods_per_node(); see ThFuelRods.h
    double th_relaxation        = 1.0;
    double h_table_max          = 0.0;

    /// HOST inputs.  Null when the corresponding device buffer is being
    /// borrowed instead (see `xskf_device` / `phif_device` below).
    const double* xskf = nullptr; ///< [ig * nxyz + l]
    const double* phif = nullptr; ///< [l * ng + ig]
    const int*    burn = nullptr; ///< [nxyz], the integer burnup key

    /// The SAME two blocks as DEVICE addresses, when a canonical owner already
    /// holds them, plus the event that orders this backend's stream behind the
    /// solve that wrote them.  ALL OR NOTHING, AND THE CALLER OWNS THE
    /// OWNERSHIP CHECK: XSSet fills these only when the canonical state says the
    /// device wrote them last.  A null pair is not an error, it is "use the host
    /// copy" -- which is what a stub build or a declined solve gives.
    const void* xskf_device  = nullptr;
    const void* phif_device  = nullptr;
    void*       device_ready = nullptr; ///< cudaEvent_t, or null

    /// HOST outputs, read AND written: the arm blends against the values these
    /// hold on entry, exactly as UpdateTH snapshots them before SolveTH.
    double* tful = nullptr; ///< [nxyz]
    double* tmod = nullptr; ///< [nxyz]
    double* dmod = nullptr; ///< [nxyz]
};

} // namespace thgpu

/// The device T/H backend.  Same shape as CramBackend / PprBackend, so the call
/// sites in XSSet.cpp never need an #ifdef.
class ThBackend {
public:
    ThBackend();
    ~ThBackend();

    ThBackend(const ThBackend&)            = delete;
    ThBackend& operator=(const ThBackend&) = delete;

    /// True when RASBERY_GPU_TH is set to a truthy value AND a device exists.
    [[nodiscard]] bool available() const;

    /// Human-readable reason when available() is false, or the last decline.
    [[nodiscard]] const std::string& status() const;

    /// One T/H update on the device.  Returns false without touching @p v's
    /// outputs when the arm is off, the device refused, or the deck's shape is
    /// one this kernel does not serve -- and the caller must then run
    /// XSSet::SolveTH unchanged.  On true, `delta_dop` is UpdateTH's return.
    ///
    /// FAIL OPEN, NEVER THROW.  A throw out of a kernel is not a thing, and a
    /// throw out of the backend would take down a 64-deck batch for one slot.
    bool solveTh(const thgpu::TableView& tables, const thgpu::GeomView& geom,
                 const thgpu::UpdateView& v, double& delta_dop);

    // --- receipt ----------------------------------------------------------
    [[nodiscard]] unsigned long long deviceUpdates() const;
    [[nodiscard]] unsigned long long bytesElided() const;
    [[nodiscard]] unsigned long long bytesH2d() const;
    [[nodiscard]] unsigned long long bytesD2h() const;
    [[nodiscard]] double             wallMs() const;
    [[nodiscard]] int                deviceOrdinal() const;
    /// The mask the kernels were launched with, or 0 before the first update.
    [[nodiscard]] unsigned long long formsMask() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rasbery
