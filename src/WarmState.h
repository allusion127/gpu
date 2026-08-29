#pragma once

// WP10.2 -- the compact warm-start state.
//
// WHAT IT IS FOR.  The GA evaluator plan's outer-cause census (Sec 2.2) charges
// 347 of a 4,609-outer case -- 7.5 % -- to the `initial` bucket: the cost of
// re-converging the flux on the state the deck handed over, before any feedback
// has fired.  A child candidate that differs from its parent by a few assembly
// swaps starts from a flux that is nearly right, and starting it from `1.0`
// everywhere throws that away.  This is the smallest object that can carry it.
//
// WHY NOT THE RESTART FILE.  The deck schema ALREADY has a warm-start channel:
// the `restart` block (IO.cpp) restores burnup, isotope inventory, T/H state and
// flux from a full HDF5 restart.  It is the right thing for continuing a CYCLE
// and the wrong thing for seeding a SIBLING:
//
//   * it is an HDF5 file, and HDF5 1.10.x is process-global and not
//     thread-safe, so every batch worker reading one queues behind the others
//     (GA evaluator plan Sec 3.1(a) measured what that costs);
//   * `--result light` writes no restart at all, and light IS the GA arm;
//   * it restores the isotope inventory and burnup, which for a DIFFERENT
//     loading pattern is not a warm start, it is the wrong fuel.
//
// So the warm state carries only what is a GUESS and never an input: the BOC
// scalar flux, the critical boron the parent converged to, and its k_eff.  Every
// one of those is overwritten by the solve; none of them is fuel, geometry or
// inventory.  That is what keeps the failure mode to "converges from a worse
// starting point" instead of "converges to a different problem".
//
// WHY BOC AND ONLY BOC.  The `initial` bucket is per statepoint, but only the
// FIRST statepoint of a case starts from the deck rather than from the previous
// statepoint's converged flux -- every later one already has the best warm start
// there is.  A file per statepoint would be 35x the bytes to seed the one
// statepoint that is not already seeded.
//
// GATE CLASS N1, NOT B0, AND THE HEADER SAYS SO.  A different starting point can
// select a different root where the Xe<->flux map has more than one (the i-SMR
// CY02 `primeXeDamping` precedent, A2_OUTER_REDUCTION Sec 5).  The trajectory
// digest is therefore ALLOWED to move, and the gate is that keff / CBC / Fq /
// FdH land inside the acceptance thresholds -- GA evaluator plan Sec 5.4 states
// exactly that.  With no warm start requested nothing here runs and the run is
// byte-identical to one built without this file.
//
// THE FILE IS HOST-LOCAL.  Doubles are written in host byte order and host
// representation, deliberately: this is a performance seed passed between
// processes on one machine within one campaign, not an archive format.  A file
// that does not match the reader's magic, version or shape is REFUSED, not
// reinterpreted, and the refusal is a receipt rather than an error -- a warm
// start that cannot be honoured must degrade to a cold start, never to a wrong
// one.

#include "Sha256.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace rasbery::warmstate {

inline constexpr char        kMagic[8] = {'R', 'A', 'S', 'B', 'W', 'R', 'M', '1'};
inline constexpr std::uint32_t kVersion = 1;

/// The whole of it.  Three scalars and one array, and every field is a GUESS:
/// the solve overwrites all of them.
struct State {
    std::uint32_t       ng    = 0;
    std::uint32_t       nxyz  = 0;
    std::uint32_t       nx    = 0;
    std::uint32_t       ny    = 0;
    std::uint32_t       nz    = 0;
    double              keff  = 1.0;
    double              boron = 0.0;
    double              efpd  = 0.0;
    std::vector<double> flux; ///< ng*nxyz, in Geometry's Phif layout

    [[nodiscard]] bool shapeMatches(const State& other) const {
        return ng == other.ng && nxyz == other.nxyz && nx == other.nx && ny == other.ny &&
               nz == other.nz;
    }
};

namespace detail {

template <class T>
void put(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <class T>
bool get(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(in);
}

} // namespace detail

/// Write the state.  Returns an empty string on success, the reason otherwise.
inline std::string save(const std::string& path, const State& state) {
    if (state.flux.size() != static_cast<std::size_t>(state.ng) * state.nxyz)
        return "flux size does not match ng*nxyz";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return "cannot open for writing: " + path;
    out.write(kMagic, sizeof(kMagic));
    detail::put(out, kVersion);
    detail::put(out, state.ng);
    detail::put(out, state.nxyz);
    detail::put(out, state.nx);
    detail::put(out, state.ny);
    detail::put(out, state.nz);
    detail::put(out, state.keff);
    detail::put(out, state.boron);
    detail::put(out, state.efpd);
    out.write(reinterpret_cast<const char*>(state.flux.data()),
              static_cast<std::streamsize>(state.flux.size() * sizeof(double)));
    if (!out) return "write failed: " + path;
    return {};
}

/// Read the state.  Returns an empty string on success, the reason otherwise --
/// and on any reason the caller must start COLD, which is why every refusal is
/// a string and none of them is an exception.
inline std::string load(const std::string& path, State& state) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "cannot open: " + path;
    char magic[sizeof(kMagic)] = {};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
        return "not a warm-state file: " + path;
    std::uint32_t version = 0;
    if (!detail::get(in, version)) return "truncated header: " + path;
    if (version != kVersion)
        return "warm-state version " + std::to_string(version) + " != " +
               std::to_string(kVersion);
    if (!detail::get(in, state.ng) || !detail::get(in, state.nxyz) ||
        !detail::get(in, state.nx) || !detail::get(in, state.ny) ||
        !detail::get(in, state.nz) || !detail::get(in, state.keff) ||
        !detail::get(in, state.boron) || !detail::get(in, state.efpd))
        return "truncated header: " + path;
    const std::size_t count = static_cast<std::size_t>(state.ng) * state.nxyz;
    if (count == 0 || count > (1u << 30)) return "implausible flux size in " + path;
    state.flux.assign(count, 0.0);
    in.read(reinterpret_cast<char*>(state.flux.data()),
            static_cast<std::streamsize>(count * sizeof(double)));
    if (in.gcount() != static_cast<std::streamsize>(count * sizeof(double)))
        return "truncated flux: " + path;
    return {};
}

/// The CONTENT digest of a warm-state file, for the case key's warm-start
/// provenance.  A path is not provenance: two runs can name one path and mean
/// different files.
inline std::string digest(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    Sha256              sha;
    std::vector<char>   buffer(64 * 1024);
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto got = in.gcount();
        if (got <= 0) break;
        sha.update(buffer.data(), static_cast<std::size_t>(got));
    }
    return sha.hex();
}

} // namespace rasbery::warmstate
