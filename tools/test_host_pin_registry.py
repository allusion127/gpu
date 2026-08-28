#!/usr/bin/env python3
"""HostPinLease registry contract (plan Rev.4 Sec 6).

Compiles src/HostPinRegistry.h on its own and runs the lease lifecycle against
fake register/unregister hooks -- no CUDA, no device.  Any of c++/g++/clang++
serves; on the Windows authoring host the MSVC toolchain is discovered through
vswhere and driven under vcvars64, so the compiled scenarios run there too.
Only when no compiler at all is found does it fall back to the static contract
over the header and the owner destructors.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HEADER = SRC / "HostPinRegistry.h"


HARNESS = r'''
#include "HostPinRegistry.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace rasbery;

static std::vector<std::pair<void*, std::size_t>> g_registered;
static std::vector<void*>                         g_unregistered;

static int fakeRegister(void* address, std::size_t bytes) {
    g_registered.emplace_back(address, bytes);
    return 0;
}

static int fakeUnregister(void* address) {
    g_unregistered.push_back(address);
    return 0;
}

#define CHECK(cond, code)                                                        \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond);         \
            return (code);                                                       \
        }                                                                        \
    } while (false)

// Eight pages, page-aligned, so the conflict intervals below are exact.
alignas(4096) static unsigned char g_pool[8 * 4096];

int main(int argc, char** argv) {
    const std::string scenario = argc > 1 ? argv[1] : "default";
    rasberyInstallHostPinHooks(&fakeRegister, &fakeUnregister);
    CHECK(rasberyHostPinHooksInstalled(), 1);

    unsigned char* const pool = g_pool;

    if (scenario == "off") {
        // RASBERY_HOST_PINNING=off: no registration, whatever the gate says.
        CHECK(rasberyHostPinningMode() == HostPinningMode::Off, 10);
        CHECK(std::strcmp(rasberyHostPinningModeName(), "off") == 0, 11);
        rasberySetHostPinningEnabled(true);
        CHECK(!rasberyHostPinningEnabled(), 12);
        CHECK(!rasberyPinHost(pool, 4096), 13);
        CHECK(rasberyHostPinLiveRanges() == 0, 14);
        CHECK(g_registered.empty(), 15);
        return 0;
    }

    if (scenario == "force") {
        // force overrides the programmatic gate main() would have cleared.
        CHECK(rasberyHostPinningMode() == HostPinningMode::Force, 20);
        rasberySetHostPinningEnabled(false);
        CHECK(rasberyHostPinningEnabled(), 21);
        CHECK(rasberyPinHost(pool, 4096), 22);
        CHECK(g_registered.size() == 1, 23);
        return 0;
    }

    if (scenario == "strict") {
        // RASBERY_PIN_STRICT=1 turns the overlap refusal into an error.  The
        // request has to come from a base the record does NOT belong to,
        // because a sole owner widening its OWN range is the safe upgrade
        // below, not an aliasing hazard.
        CHECK(rasberyHostPinStrict(), 30);
        CHECK(rasberyPinHost(pool + 4096, 8192, "strict.holder"), 31);
        bool threw = false;
        try {
            rasberyPinHost(pool, 4 * 4096, "strict.straddle"); // different base, wider
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, 32);
        CHECK(rasberyHostPinCounters().overlap_rejections == 1, 33);
        CHECK(rasberyHostPinCounters().upgraded_ranges == 0, 36);
        CHECK(g_registered.size() == 1, 34);   // nothing was re-registered
        CHECK(g_unregistered.empty(), 35);     // and nothing was torn down
        return 0;
    }

    if (scenario == "upgrade") {
        // SAFE UPGRADE (fix strategy b).  A wider request from the record's
        // SOLE owner is admissible: no foreign owner is evicted, and the new
        // range is a superset, so nothing is aliased.
        void* const a = pool + 4096;
        CHECK(rasberyHostPinUpgradeEnabled(), 400);
        CHECK(rasberyPinHost(a, 128, "narrow"), 401);
        CHECK(g_registered.size() == 1 && g_registered[0].second == 128, 402);
        CHECK(rasberyHostPinOwners(a) == 1, 403);

        CHECK(rasberyPinHost(a, 3 * 4096, "wide"), 404);
        CHECK(rasberyHostPinCounters().upgraded_ranges == 1, 405);
        CHECK(rasberyHostPinCounters().overlap_rejections == 0, 406);
        // torn down at the address cudaHostRegister saw, retaken at the new width
        CHECK(g_unregistered.size() == 1 && g_unregistered[0] == a, 407);
        CHECK(g_registered.size() == 2 && g_registered[1].first == a &&
                  g_registered[1].second == 3 * 4096,
              408);
        CHECK(rasberyHostPinOwners(a) == 1, 409);
        // the register/unregister ledger still balances against the live record
        CHECK(rasberyHostPinCounters().registered_ranges == 2, 410);
        CHECK(rasberyHostPinCounters().unregistered_ranges == 1, 411);
        // and the widened range really is the one now covered
        CHECK(rasberyPinHost(pool + 3 * 4096, 64, "inside-the-upgrade"), 412);
        CHECK(rasberyHostPinOwners(a) == 2, 413);

        // A SECOND owner blocks the next upgrade: widening would tear down a
        // range that other owner is entitled to (Sec 6.3 is not negotiable).
        CHECK(!rasberyPinHost(a, 6 * 4096, "wider-with-foreign-owner"), 414);
        CHECK(rasberyHostPinCounters().overlap_rejections == 1, 415);
        CHECK(rasberyHostPinCounters().upgraded_ranges == 1, 416);
        CHECK(g_unregistered.size() == 1, 417); // nothing torn down

        // A wider request from a DIFFERENT base is never an upgrade either,
        // even once it is the only other owner that is gone.
        rasberyUnpinHost(pool + 3 * 4096);
        CHECK(rasberyHostPinOwners(a) == 1, 418);
        CHECK(!rasberyPinHost(pool, 8 * 4096, "different-base"), 419);
        CHECK(rasberyHostPinCounters().overlap_rejections == 2, 420);
        CHECK(g_unregistered.size() == 1, 421);

        rasberyUnpinHost(a);
        CHECK(rasberyHostPinLiveRanges() == 0, 422);
        CHECK(g_unregistered.size() == 2, 423);
        return 0;
    }

    if (scenario == "upgrade_off") {
        // RASBERY_PIN_UPGRADE=0 puts the wider-request case back on the plain
        // Sec 6.2 refusal, unchanged from the pre-upgrade registry.
        CHECK(!rasberyHostPinUpgradeEnabled(), 430);
        void* const a = pool + 4096;
        CHECK(rasberyPinHost(a, 128, "narrow"), 431);
        CHECK(!rasberyPinHost(a, 3 * 4096, "wide"), 432);
        CHECK(rasberyHostPinCounters().overlap_rejections == 1, 433);
        CHECK(rasberyHostPinCounters().upgraded_ranges == 0, 434);
        CHECK(g_unregistered.empty(), 435);
        return 0;
    }

    if (scenario == "canonical") {
        // The SOURCE fix.  Page-exclusive storage is what keeps two DIFFERENT
        // buffers out of each other's pages: 5000 bytes is deliberately not a
        // page multiple, so a general-purpose allocator would start the next
        // block inside this one's last page and every registration after the
        // first would be refused.
        constexpr int         kBuffers = 32;
        constexpr std::size_t kCount   = 625; // 5000 bytes
        constexpr std::size_t kBytes   = kCount * sizeof(double);
        std::vector<double*>  buffers;
        for (int i = 0; i < kBuffers; ++i)
            buffers.push_back(rasberyPageExclusiveZeroedArray<double>(kCount));

        for (int i = 0; i < kBuffers; ++i) {
            CHECK(buffers[i] != nullptr, 500);
            CHECK(buffers[i][0] == 0.0 && buffers[i][kCount - 1] == 0.0, 501);
            CHECK(reinterpret_cast<std::uintptr_t>(buffers[i]) % 32 == 0, 502);
            CHECK(rasberyPinHost(buffers[i], kBytes, "canonical"), 503);
        }
        CHECK(rasberyHostPinCounters().overlap_rejections == 0, 504);
        CHECK(rasberyHostPinCounters().pageable_fallbacks == 0, 505);
        CHECK(rasberyHostPinLiveRanges() == static_cast<std::size_t>(kBuffers), 506);

        // Page intervals are pairwise disjoint -- that IS the property, stated
        // directly rather than inferred from the counters above.
        const auto page_begin = [](const void* p) {
            return reinterpret_cast<std::uintptr_t>(p) & ~(kHostPinPageSize - 1);
        };
        const auto page_end = [](const void* p, std::size_t bytes) {
            return (reinterpret_cast<std::uintptr_t>(p) + bytes + kHostPinPageSize - 1) &
                   ~(kHostPinPageSize - 1);
        };
        for (int i = 0; i < kBuffers; ++i)
            for (int j = i + 1; j < kBuffers; ++j)
                CHECK(page_end(buffers[i], kBytes) <= page_begin(buffers[j]) ||
                          page_end(buffers[j], kBytes) <= page_begin(buffers[i]),
                      507);

        // CANONICAL WIDTH: a repeat request at the same base and the same width
        // deduplicates, and a consumer that only needs a sub-range asks from an
        // interior base and becomes a second owner.  Neither is a refusal.
        CHECK(rasberyPinHost(buffers[0], kBytes, "canonical.repeat"), 508);
        CHECK(rasberyHostPinCounters().deduplicated_requests == 1, 509);
        CHECK(rasberyPinHost(buffers[0] + 8, 64, "canonical.subrange"), 510);
        CHECK(rasberyHostPinOwners(buffers[0]) == 2, 511);
        CHECK(rasberyHostPinCounters().overlap_rejections == 0, 512);

        rasberyUnpinHost(buffers[0] + 8);
        for (int i = 0; i < kBuffers; ++i) rasberyUnpinHost(buffers[i]);
        CHECK(rasberyHostPinLiveRanges() == 0, 513);
        CHECK(rasberyHostPinCounters().registered_ranges ==
                  rasberyHostPinCounters().unregistered_ranges,
              514);
        for (int i = 0; i < kBuffers; ++i) rasberyPageExclusiveDeleteArray(buffers[i]);

        // The std::vector flavour the sweep staging buffers use.
        PageExclusiveVector<double> staged(kCount, 1.0);
        CHECK(rasberyPinHost(staged.data(), staged.size() * sizeof(double), "canonical.vector"),
              515);
        CHECK(rasberyHostPinCounters().overlap_rejections == 0, 516);
        rasberyUnpinHost(staged.data());
        CHECK(rasberyHostPinLiveRanges() == 0, 517);
        return 0;
    }

    if (scenario == "debug") {
        // RASBERY_PIN_DEBUG=1 must name BOTH call sites on the rejection line;
        // the Python driver greps stderr for the pair.
        CHECK(rasberyHostPinDebug(), 600);
        CHECK(rasberyPinHost(pool + 4096, 8192, "site.A"), 601);
        CHECK(!rasberyPinHost(pool, 4 * 4096, "site.B"), 602);
        CHECK(rasberyHostPinCounters().overlap_rejections == 1, 603);
        return 0;
    }

    // ---- default: the whole lease lifecycle -------------------------------
    CHECK(rasberyHostPinningMode() == HostPinningMode::Auto, 100);
    CHECK(rasberyHostPinningEnabled(), 101);

    // 1. fresh registration, at the caller's address and byte count (Sec 6.1).
    void* const a = pool + 4096;
    CHECK(rasberyPinHost(a, 8192), 110);
    CHECK(g_registered.size() == 1, 111);
    CHECK(g_registered[0].first == a && g_registered[0].second == 8192, 112);
    CHECK(rasberyHostPinLiveRanges() == 1, 113);
    CHECK(rasberyHostPinOwners(a) == 1, 114);
    CHECK(rasberyHostPinCounters().registered_ranges == 1, 115);
    CHECK(rasberyHostPinCounters().registered_bytes == 8192, 116);

    // 2. the same base again is idempotent: one buffer, one owner.  This is the
    //    Phif case -- four callers page-lock it, ~Geometry releases it once.
    CHECK(rasberyPinHost(a, 8192), 120);
    CHECK(g_registered.size() == 1, 121);
    CHECK(rasberyHostPinOwners(a) == 1, 122);
    CHECK(rasberyHostPinCounters().deduplicated_requests == 1, 123);

    // 3. requested SUBSET existing from a DIFFERENT base -> second owner.
    void* const b = pool + 4096 + 128;
    CHECK(rasberyPinHost(b, 256), 130);
    CHECK(g_registered.size() == 1, 131);
    CHECK(rasberyHostPinOwners(a) == 2, 132);
    CHECK(rasberyHostPinCounters().deduplicated_requests == 2, 133);

    // 4. existing SUBSET requested -> pageable fallback, never an expand.  The
    //    safe upgrade cannot apply here: b is a second owner, and widening
    //    would tear down a range b is entitled to.
    CHECK(!rasberyPinHost(a, 4 * 4096), 140);
    CHECK(rasberyHostPinCounters().overlap_rejections == 1, 141);
    CHECK(rasberyHostPinCounters().pageable_fallbacks == 1, 142);
    CHECK(rasberyHostPinCounters().upgraded_ranges == 0, 145);
    CHECK(g_registered.size() == 1 && g_unregistered.empty(), 143);
    CHECK(rasberyHostPinOwners(a) == 2, 144);

    // 5. partial overlap (starts before the record, ends inside it) -> refused.
    CHECK(!rasberyPinHost(pool, 4096 + 8), 150);
    CHECK(rasberyHostPinCounters().overlap_rejections == 2, 151);
    CHECK(g_registered.size() == 1 && g_unregistered.empty(), 152);

    // 6. releasing an address that never held a lease is a no-op -- including
    //    the one just refused, whose page falls inside the live record.
    rasberyUnpinHost(pool);
    CHECK(rasberyHostPinOwners(a) == 2, 160);
    CHECK(g_unregistered.empty(), 161);

    // 7. release: unregister only when the LAST owner is gone (Sec 6.3), at the
    //    original registered address.
    rasberyUnpinHost(b);
    CHECK(rasberyHostPinOwners(a) == 1, 170);
    CHECK(g_unregistered.empty(), 171);
    rasberyUnpinHost(a);
    CHECK(rasberyHostPinLiveRanges() == 0, 172);
    CHECK(g_unregistered.size() == 1 && g_unregistered[0] == a, 173);
    CHECK(rasberyHostPinCounters().unregistered_ranges == 1, 174);

    // 8. the recycled-worker case: the same address, leased again after the
    //    previous tenant released it, registers AFRESH instead of aliasing.
    CHECK(rasberyPinHost(a, 8192), 180);
    CHECK(g_registered.size() == 2 && g_registered[1].first == a, 181);
    rasberyUnpinHost(a);
    CHECK(rasberyHostPinLiveRanges() == 0, 182);

    // 9. stale eviction: a record whose owner never released it (owners == 0)
    //    is evicted when a new request overlaps it -- and only then.
    void* const c = pool + 6 * 4096;
    {
        std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
        PinRecord                   stray;
        stray.conflict_page_begin = reinterpret_cast<std::uintptr_t>(c);
        stray.conflict_page_end   = stray.conflict_page_begin + 4096;
        stray.registered_address  = c;
        stray.registered_bytes    = 4096;
        stray.owners              = 0;
        rasberyHostPinRecords().emplace(stray.conflict_page_begin, stray);
    }
    const unsigned long long unregistered_before =
        rasberyHostPinCounters().unregistered_ranges;
    CHECK(rasberyPinHost(c, 2048), 190);
    CHECK(rasberyHostPinCounters().stale_evicted == 1, 191);
    CHECK(rasberyHostPinCounters().unregistered_ranges == unregistered_before + 1, 192);
    CHECK(rasberyHostPinOwners(c) == 1, 193);

    // 10. drain: owners > 0 is reported, not torn down; owners == 0 is drained.
    rasberyDrainPinnedRegistry();
    CHECK(rasberyHostPinLiveRanges() == 1, 200); // c is still leased
    rasberyUnpinHost(c);
    CHECK(rasberyHostPinLiveRanges() == 0, 201);
    {
        std::lock_guard<std::mutex> lock(rasberyHostPinMutex());
        PinRecord                   stray;
        stray.conflict_page_begin = reinterpret_cast<std::uintptr_t>(pool + 7 * 4096);
        stray.conflict_page_end   = stray.conflict_page_begin + 4096;
        stray.registered_address  = pool + 7 * 4096;
        stray.registered_bytes    = 4096;
        stray.owners              = 0;
        rasberyHostPinRecords().emplace(stray.conflict_page_begin, stray);
    }
    rasberyDrainPinnedRegistry();
    CHECK(rasberyHostPinLiveRanges() == 0, 202);
    CHECK(g_unregistered.back() == pool + 7 * 4096, 203);

    // 11. the Sec 6.8 receipt carries every field the parser requires.
    return 0;
}
'''


STATIC_TOKENS = {
    "src/HostPinRegistry.h": (
        "struct PinRecord",
        "conflict_page_begin",
        "conflict_page_end",
        "registered_address",
        "registered_bytes",
        "unsigned owners",
        "unsigned in_flight",
        "HostPinRegisterHook",
        "HostPinUnregisterHook",
        "rasberyInstallHostPinHooks",
        "inline bool rasberyPinHost",
        "inline void rasberyUnpinHost",
        "inline void rasberyDrainPinnedRegistry",
        "RASBERY_HOST_PINNING",
        "RASBERY_PIN_STRICT",
        "HostPinningMode::Force",
        "deduplicated_requests",
        "pageable_fallbacks",
        "unregistered_ranges",
        "overlap_rejections",
        "stale_evicted",
        "upgraded_ranges",
        "RASBERY_PIN_DEBUG",
        "RASBERY_PIN_UPGRADE",
        "logRejectionLocked",
        # The source fix: page-exclusive storage for everything that gets pinned.
        "rasberyPageExclusiveAlloc",
        "rasberyPageExclusiveFree",
        "rasberyPageExclusiveZeroedArray",
        "rasberyPageExclusiveDeleteArray",
        "PageExclusiveVector",
        "\\\"pinning_mode\\\":\\\"",
    ),
    # Every buffer the backends page-lock owns its pages outright, so no two
    # registrations can collide on a shared boundary page.
    "include/milk.h": (
        "kPageExclusiveThreshold",
        "isPageExclusive",
        "nextPageSkew",
        "deallocate(T* ptr, std::size_t count)",
    ),
    # Every owner releases in its destructor, before the memory is freed.
    "src/Geometry.cpp": (
        "rasberyUnpinHost(_phif);",
        "rasberyUnpinHost(_jnet);",
        "rasberyUnpinHost(_phis);",
        "_phif = rasberyPageExclusiveZeroedArray<double>(",
        "rasberyPageExclusiveDeleteArray(_jnet);",
        # _vol is page-locked as `geom.vol@sweep` since BICGCMFD stopped
        # staging a private copy of it (it aliases &_g.vol(0) now), so it needs
        # BOTH halves of the owner contract: page-exclusive storage, or the
        # registration is refused as an overlap and the buffer quietly falls
        # back to pageable; and a release in ~Geometry, or the next deck on a
        # recycled worker inherits a live registration at its own addresses.
        "rasberyUnpinHost(_vol);",
        "_vol = rasberyPageExclusiveZeroedArray<double>(",
        "rasberyPageExclusiveDeleteArray(_vol);",
    ),
    "src/Nodal.cpp": (
        "rasberyUnpinHost(_eta1);",
        "rasberyUnpinHost(_diagDI);",
        "_eta1    = rasberyPageExclusiveArray<double>(nconst);",
        "rasberyPageExclusiveDeleteArray(_diagD);",
    ),
    "src/CMFD.cpp": (
        "rasberyUnpinHost(_dtil);",
        "rasberyUnpinHost(_psi);",
        "_dtil  = rasberyPageExclusiveZeroedArray<double>(",
        "rasberyPageExclusiveDeleteArray(_psi);",
    ),
    "src/BICGCMFD.cpp": (
        "_ls.reset();",
        "rasberyUnpinHost(_pin_udiag);",
        "rasberyUnpinHost(_pin_sweep_vol);",
        "lease_vector(",
        # every sweep-path request carries its call-site tag
        '"cmfd.diag@sweep"',
        '"xs.xssm@sweep"',
        # xsnf / vol / chif are ALIASED into the sweep from their owners now
        # (XSSet's _xs.xsnf and _ref_chix, Geometry's _vol), so they are
        # page-locked under the OWNER's tag and released by the owner's
        # destructor -- a lease_vector here would release the only owner record
        # for a base XSSet/Geometry still uses.  The BICGCMFD-owned chif
        # fallback (a deck with no fission spectrum) keeps its own tag.
        '"xs.xsnf@sweep"',
        '"geom.vol@sweep"',
        '"bicg.sweep_chif@sweep"',
    ),
    "src/BICGCMFD.h": (
        "PageExclusiveVector<double> _udiag;",
        "PageExclusiveVector<double> _sweep_chif, _sweep_xsnf, _sweep_vol;",
    ),
    "src/XSSet.h": (
        "PageExclusiveVector<double> _ref_chix;",
    ),
    "src/XSSet.cpp": (
        "_xsrecon_backend.reset();",
        "rasberyUnpinHost(_iden.data());",
        "rasberyUnpinHost(_ref_micx.xssm.data());",
        # chifData(), page-locked by NodalArena::pinSlot and owned here.
        "rasberyUnpinHost(_ref_chix.data());",
        # both arms ask for the SAME bases at the SAME widths, tagged per arm
        '"xs.micx@flatxs"',
        '"xs.micx@xsrecon"',
        '"geom.phif@xsrecon"',
    ),
    "src/CudaXsReconBackend.cu": (
        '"geom.jnet@arena"',
        '"nodal.const@arena"',
        '"xs.chif@arena"',
        '"geom.phif@nodal"',
    ),
    "src/main.cpp": (
        "rasberyDrainPinnedRegistry();",
        "rasberyAppendHostPinReceiptFields(std::cout);",
        "rasbery::HostPinningMode::Off",
    ),
}


# Spellings that MUST NOT come back.  Page-exclusive storage and `new[]` are
# not interchangeable: rasberyPageExclusiveAlloc hands out a pointer SKEWED
# into its block, so `delete[]` on it frees an address the allocator never
# issued, and a plain `new double[]` for a buffer the backends page-lock puts
# it back on a shared boundary page where cudaHostRegister refuses it.  Both
# failures are silent -- one is heap corruption nobody sees until later, the
# other is a performance cliff that only shows up as overlap_rejections in the
# receipt -- so they are asserted absent rather than looked for by eye.
# Matched as REGEXES, so a reintroduction cannot slip through on whitespace.
FORBIDDEN_PATTERNS = {
    "src/Geometry.cpp": (
        (r"delete\s*\[\s*\]\s*_vol\s*;",
         "_vol is page-exclusive; free it with rasberyPageExclusiveDeleteArray"),
        (r"_vol\s*=\s*new\s+double\s*\[",
         "_vol is page-locked as geom.vol@sweep and must own its pages"),
    ),
}


def static_contract() -> list[str]:
    problems: list[str] = []
    for relative, tokens in STATIC_TOKENS.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                problems.append(f"{relative}: missing {token!r}")
    for relative, entries in FORBIDDEN_PATTERNS.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for pattern, why in entries:
            found = re.search(pattern, text)
            if found:
                problems.append(f"{relative}: {found.group(0)!r} must be gone -- {why}")
    # The permanent-registration contract must be gone from both pin bodies.
    for relative in ("src/CudaXsReconBackend.cu", "src/CudaBICGBackend.cu"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "return rasberyPinHost(p, bytes, tag);" not in text:
            problems.append(f"{relative}: pinHost does not go through the registry")
        if "installHostPinHooks();" not in text:
            problems.append(f"{relative}: CUDA register/unregister hooks are never installed")
    return problems


# Every environment override the compiled scenarios read, cleared before each
# run so the parent shell cannot leak one in.
PIN_ENV = ("RASBERY_HOST_PINNING", "RASBERY_PIN_STRICT", "RASBERY_PIN_DEBUG",
           "RASBERY_PIN_UPGRADE")

SCENARIOS = (
    ("default", {}),
    ("off", {"RASBERY_HOST_PINNING": "off"}),
    ("force", {"RASBERY_HOST_PINNING": "force"}),
    ("strict", {"RASBERY_PIN_STRICT": "1"}),
    ("upgrade", {}),
    ("upgrade_off", {"RASBERY_PIN_UPGRADE": "0"}),
    ("canonical", {}),
    ("debug", {"RASBERY_PIN_DEBUG": "1"}),
)


def msvc_vcvars() -> str | None:
    """vcvars64.bat of the newest MSVC install, or None off Windows."""
    if os.name != "nt":
        return None
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    done = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, universal_newlines=True)
    root = done.stdout.strip().splitlines()
    if done.returncode != 0 or not root:
        return None
    bat = Path(root[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return str(bat) if bat.is_file() else None


def compile_harness(compiler: str, cpp: Path, exe: Path) -> None:
    """Build the harness with either a POSIX compiler or MSVC under vcvars64."""
    if compiler.lower().endswith("vcvars64.bat"):
        # cl.exe only has its include/lib environment under vcvars, and it drops
        # its intermediates in the CWD, so the build runs from a batch file inside
        # the temp directory -- a script sidesteps cmd.exe's quoting rules for the
        # space-bearing Visual Studio path.  /WX because a warning in this header
        # is a contract break too.
        script = cpp.parent / "build_host_pin_registry_test.bat"
        script.write_text(
            "@echo off\r\n"
            + 'call "%s" >nul\r\n' % compiler
            + 'cl /nologo /std:c++20 /EHsc /W4 /WX /D_CRT_SECURE_NO_WARNINGS '
              '/I "%s" "%s" /Fe:"%s"\r\n' % (SRC, cpp, exe),
            encoding="utf-8")
        subprocess.run(["cmd", "/c", str(script)], check=True, cwd=str(cpp.parent),
                       capture_output=True, universal_newlines=True)
        return
    subprocess.run(
        [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
         "-I", str(SRC), str(cpp), "-o", str(exe)],
        check=True,
    )


def run_harness(compiler: str) -> list[str]:
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cpp = tmp_path / "host_pin_registry_test.cpp"
        exe = tmp_path / ("host_pin_registry_test.exe" if os.name == "nt"
                          else "host_pin_registry_test")
        cpp.write_text(HARNESS, encoding="utf-8")
        try:
            compile_harness(compiler, cpp, exe)
        except subprocess.CalledProcessError as failure:
            output = (failure.stdout or "") + (failure.stderr or "")
            return ["harness did not compile: " + output.strip()[-2000:]]
        for name, overrides in SCENARIOS:
            env = os.environ.copy()
            for key in PIN_ENV:
                env.pop(key, None)
            env.update(overrides)
            done = subprocess.run([str(exe), name], env=env, capture_output=True,
                                  universal_newlines=True)
            if done.returncode != 0:
                problems.append(
                    "scenario %s exited %d: %s" % (name, done.returncode, done.stderr.strip())
                )
                continue
            if name == "debug":
                # The rejection line is the diagnostic this whole exercise turns
                # on: it has to carry both call-site tags and both byte counts.
                line = next((l for l in done.stderr.splitlines()
                             if "[RASBERY][PIN][reject]" in l), "")
                for needed in ('"site.B"', '"site.A"', '"bytes":16384', '"bytes":8192',
                               '"owners":1'):
                    if needed not in line:
                        problems.append(
                            "RASBERY_PIN_DEBUG rejection line lacks %s: %r" % (needed, line))
    return problems


def main() -> int:
    problems = static_contract()
    compiler = (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
                or msvc_vcvars())
    if compiler is not None:
        problems.extend(run_harness(compiler))
    if problems:
        for problem in problems:
            print("host pin registry: FAIL " + problem, file=sys.stderr)
        return 1
    if compiler is None:
        print("host pin registry: static contract PASS "
              "(no C++ compiler here -- the compiled harness was skipped)")
    else:
        print("host pin registry: PASS (static contract + %d compiled scenarios, %s)"
              % (len(SCENARIOS), Path(compiler).name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
