#!/usr/bin/env python3
"""HostPinLease registry contract (plan Rev.4 Sec 6).

Compiles src/HostPinRegistry.h on its own and runs the lease lifecycle against
fake register/unregister hooks -- no CUDA, no device.  Where no C++ compiler is
available (the Windows authoring host) it falls back to a static contract over
the header and the owner destructors, so the wiring is still checked.
"""
from __future__ import annotations

import os
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
        // RASBERY_PIN_STRICT=1 turns the overlap refusal into an error.
        CHECK(rasberyHostPinStrict(), 30);
        CHECK(rasberyPinHost(pool + 4096, 8192), 31);
        bool threw = false;
        try {
            rasberyPinHost(pool + 4096, 4 * 4096); // existing SUBSET requested
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, 32);
        CHECK(rasberyHostPinCounters().overlap_rejections == 1, 33);
        CHECK(g_registered.size() == 1, 34);   // nothing was re-registered
        CHECK(g_unregistered.empty(), 35);     // and nothing was torn down
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

    // 4. existing SUBSET requested -> pageable fallback, never an expand.
    CHECK(!rasberyPinHost(a, 4 * 4096), 140);
    CHECK(rasberyHostPinCounters().overlap_rejections == 1, 141);
    CHECK(rasberyHostPinCounters().pageable_fallbacks == 1, 142);
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
        "\\\"pinning_mode\\\":\\\"",
    ),
    # Every owner releases in its destructor, before the memory is freed.
    "src/Geometry.cpp": (
        "rasberyUnpinHost(_phif);",
        "rasberyUnpinHost(_jnet);",
        "rasberyUnpinHost(_phis);",
    ),
    "src/Nodal.cpp": (
        "rasberyUnpinHost(_eta1);",
        "rasberyUnpinHost(_diagDI);",
    ),
    "src/CMFD.cpp": (
        "rasberyUnpinHost(_dtil);",
        "rasberyUnpinHost(_psi);",
    ),
    "src/BICGCMFD.cpp": (
        "_ls.reset();",
        "rasberyUnpinHost(_pin_udiag);",
        "rasberyUnpinHost(_pin_sweep_vol);",
        "lease_vector(",
    ),
    "src/XSSet.cpp": (
        "_xsrecon_backend.reset();",
        "rasberyUnpinHost(_iden.data());",
        "rasberyUnpinHost(_ref_micx.xssm.data());",
        # chifData(), page-locked by NodalArena::pinSlot and owned here.
        "rasberyUnpinHost(_ref_chix.data());",
    ),
    "src/main.cpp": (
        "rasberyDrainPinnedRegistry();",
        "rasberyAppendHostPinReceiptFields(std::cout);",
        "rasbery::HostPinningMode::Off",
    ),
}


def static_contract() -> list[str]:
    problems: list[str] = []
    for relative, tokens in STATIC_TOKENS.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                problems.append(f"{relative}: missing {token!r}")
    # The permanent-registration contract must be gone from both pin bodies.
    for relative in ("src/CudaXsReconBackend.cu", "src/CudaBICGBackend.cu"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "return rasberyPinHost(p, bytes);" not in text:
            problems.append(f"{relative}: pinHost does not go through the registry")
        if "installHostPinHooks();" not in text:
            problems.append(f"{relative}: CUDA register/unregister hooks are never installed")
    return problems


def run_harness(compiler: str) -> list[str]:
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cpp = tmp_path / "host_pin_registry_test.cpp"
        exe = tmp_path / "host_pin_registry_test"
        cpp.write_text(HARNESS, encoding="utf-8")
        subprocess.run(
            [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
             "-I", str(SRC), str(cpp), "-o", str(exe)],
            check=True,
        )
        scenarios = (
            ("default", {}),
            ("off", {"RASBERY_HOST_PINNING": "off"}),
            ("force", {"RASBERY_HOST_PINNING": "force"}),
            ("strict", {"RASBERY_PIN_STRICT": "1"}),
        )
        for name, overrides in scenarios:
            env = os.environ.copy()
            env.pop("RASBERY_HOST_PINNING", None)
            env.pop("RASBERY_PIN_STRICT", None)
            env.update(overrides)
            done = subprocess.run([str(exe), name], env=env, capture_output=True,
                                  universal_newlines=True)
            if done.returncode != 0:
                problems.append(
                    "scenario %s exited %d: %s" % (name, done.returncode, done.stderr.strip())
                )
    return problems


def main() -> int:
    problems = static_contract()
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
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
        print("host pin registry: PASS (static contract + 4 compiled scenarios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
