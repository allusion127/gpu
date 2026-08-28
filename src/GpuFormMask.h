#pragma once

// Runtime override + receipt for the mined contraction masks -- Rev.7.1 Task 5
// and Task 4 follow-up (host portability).
//
// WHY THIS EXISTS.  NodalConstantKernel.h and CmfdOuterKernel.h each carry a
// mask recording which multiply-adds THE HOST COMPILER FUSED, so the device
// build can reproduce them.  That is a property of the build host, not of the
// physics:
//
//     WSL2 / g++ 13.3 dev box   CMFD_OUTER_FORMS = 0x6
//     238 / Xeon Gold 5317      CMFD_OUTER_FORMS = 0x7
//
// A single baked constant therefore cannot serve both, and on the host it does
// not match, the Class B0 gates fail for a reason that has nothing to do with
// the code under test.  Two things follow, and this header is both of them:
//
//   1. THE TESTS MINE FIRST.  They derive the host's mask, assert against THAT,
//      and print it.  They fail when the mining is unstable, not when it
//      disagrees with a literal from another machine.
//   2. THE RUNTIME CAN BE TOLD.  A binary built on one host and validated
//      against a reference produced on another needs to be able to say so
//      without a rebuild.
//
// HOST-ONLY BY CONSTRUCTION.  getenv has no device implementation, so the
// override is resolved once on the host and the resulting value is passed into
// kernels as an ARGUMENT.  Device code never calls anything here; it receives
// `forms` and uses it.  The pure bodies already take the mask as a parameter
// precisely so this is possible.

#include <cstdlib>
#include <iostream>
#include <string>

namespace rasbery::gpu {

/// Parse a mask override.  Accepts "0x7", "7", "0X7"; anything unparseable is
/// refused LOUDLY rather than silently falling back, because a typo in this
/// variable would otherwise look like a passing run against the wrong contract.
inline bool parseFormMask(const char* text, unsigned long long& out) {
    if (text == nullptr) return false;
    const std::string s(text);
    if (s.empty()) return false;
    char*                    end   = nullptr;
    const unsigned long long value = std::strtoull(s.c_str(), &end, 0);
    if (end == nullptr || *end != '\0') return false;
    out = value;
    return true;
}

/// Resolve one mask: the per-build default unless the environment overrides it.
/// Emits ONE receipt line per mask so a run's log says which contraction
/// contract it actually ran under -- without that, an override is invisible in
/// exactly the situation where it matters.
inline unsigned long long resolveFormMask(const char* env_name,
                                          unsigned long long build_default,
                                          const char* mask_name) {
    unsigned long long value  = build_default;
    const char*        raw    = std::getenv(env_name);
    const char*        source = "build_default";
    if (raw != nullptr) {
        unsigned long long parsed = 0;
        if (parseFormMask(raw, parsed)) {
            value  = parsed;
            source = "env";
        } else {
            std::cerr << "[RASBERY][WARN][FORMS] " << env_name << "=\"" << raw
                      << "\" is not a number; using the build default instead. "
                         "A malformed override must not silently pass for a valid one.\n";
            source = "env_rejected";
        }
    }
    std::cerr << "[RASBERY][FORMS] {\"mask\":\"" << mask_name << "\",\"value\":\"0x"
              << std::hex << value << std::dec << "\",\"source\":\"" << source
              << "\",\"build_default\":\"0x" << std::hex << build_default << std::dec
              << "\"}" << std::endl;
    return value;
}

} // namespace rasbery::gpu
