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

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace rasbery::gpu {

// ---------------------------------------------------------------------------
// RASBERY_FORMS_PIN -- one switch over every contraction channel
// ---------------------------------------------------------------------------
//
// WHY IT EXISTS.  Every calibrated resolver below prefers the value it MINED on
// this host over the value the build baked, and that is the right default: the
// mined mask is a measurement of THIS binary on THIS machine.  But it makes the
// arithmetic a property of the host, so a reference produced on 238 and a rerun
// on 181 can legitimately disagree while both are "correct" -- and until this
// knob there was no way to ask for the SHIPPED contraction on both, short of
// typing five per-channel overrides whose correct values differ per channel.
//
// `RASBERY_FORMS_PIN=default` forces the build default on every channel and
// says so in the receipt (source "pinned_default").  `=mined` is the existing
// behaviour, spelled out loud.  Unset is `mined`.
//
// RANKED BELOW THE PER-MASK OVERRIDES, deliberately: a human who typed
// RASBERY_XE_FORMS=0xadd asked for that channel specifically, and a blanket pin
// must not quietly overrule a specific request.  The precedence is therefore
//
//     per-mask env override  >  RASBERY_FORMS_PIN=default  >  mined  >  default
//
// and the receipt names which of the four each channel took.
enum class FormsPin { Mined, Default };

inline FormsPin formsPin() {
    static const FormsPin resolved = [] {
        const char* raw = std::getenv("RASBERY_FORMS_PIN");
        if (raw == nullptr) return FormsPin::Mined;
        std::string s(raw);
        // Trim then case-fold, the same rule every other RASBERY mode switch
        // uses -- a CRLF-terminated env file must not turn "default" into an
        // unrecognised word and buy the opposite arm.
        constexpr const char*        kBlank = " \t\r\n\v\f";
        const std::string::size_type first  = s.find_first_not_of(kBlank);
        if (first == std::string::npos)
            s.clear();
        else
            s = s.substr(first, s.find_last_not_of(kBlank) - first + 1);
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s.empty() || s == "mined") return FormsPin::Mined;
        if (s == "default") return FormsPin::Default;
        std::cerr << "[RASBERY][WARN][FORMS] RASBERY_FORMS_PIN=\"" << raw
                  << "\" is not a mode (default|mined); using mined.  A typo must not "
                     "silently pass for a pin.\n";
        return FormsPin::Mined;
    }();
    return resolved;
}

inline const char* formsPinName() {
    return formsPin() == FormsPin::Default ? "default" : "mined";
}

// ---------------------------------------------------------------------------
// THE RESOLVED-MASK REGISTRY
// ---------------------------------------------------------------------------
//
// THE HOLE IT CLOSES.  A contraction mask selects the ROUNDING of production
// arithmetic, so two runs whose masks differ are two different trajectories --
// and until now the case key could not tell them apart.  The per-mask env
// overrides are on trajectory::kArmEnv, so an OVERRIDDEN mask was keyed; the
// value a host MINED was not, and neither was RASBERY_FORMS_PIN.  A reference
// that mined 0x7 on 238 and a rerun that mined 0x6 on the dev box therefore
// shared one case key while running two different arithmetics.
//
// Every resolver records what it resolved here, once, and casekey::Provenance
// folds the digest of the whole registry (`forms_digest`).
//
// ORDER.  The registry keeps INSERTION order for the receipt -- the order the
// channels came up is worth reading -- and digests a NAME-SORTED copy, because
// the order channels happen to resolve in is a scheduling fact and must not be
// able to move a key.
//
// FREEZING.  `formsPayloadFrozen()` latches the payload the first time it is
// asked for, which is when the first case key is computed, so that a long-lived
// process (the evaluator server) cannot compute case 1's key over an empty
// registry and case 2's over a full one.  A channel that resolves after the
// freeze is recorded and WARNED about rather than silently folded.
struct FormMaskRecord {
    std::string        name;
    unsigned long long value       = 0;
    std::string        source;
    bool               mined_sound = false;
};

namespace forms_detail {
inline std::mutex& registryMutex() {
    static std::mutex cell;
    return cell;
}
inline std::vector<FormMaskRecord>& registryCell() {
    static std::vector<FormMaskRecord> cell;
    return cell;
}
inline bool& frozenCell() {
    static bool cell = false;
    return cell;
}
/// "0x" + lowercase hex, spelled here rather than through a stream so the
/// payload bytes cannot depend on an iostream's sticky flags.
inline std::string hex(unsigned long long v) {
    static const char* digits = "0123456789abcdef";
    std::string        out;
    do {
        out.insert(out.begin(), digits[v & 0xfull]);
        v >>= 4;
    } while (v != 0);
    return "0x" + out;
}
/// The canonical bytes, one line per channel, name-sorted.  Caller holds the
/// lock.
inline std::string payloadLocked() {
    std::vector<FormMaskRecord> sorted = registryCell();
    std::sort(sorted.begin(), sorted.end(),
              [](const FormMaskRecord& a, const FormMaskRecord& b) { return a.name < b.name; });
    std::string out;
    for (const FormMaskRecord& r : sorted) {
        out += "form\t";
        out += r.name;
        out += '\t';
        out += hex(r.value);
        out += '\t';
        out += r.source;
        out += '\t';
        out += r.mined_sound ? '1' : '0';
        out += '\n';
    }
    return out;
}
} // namespace forms_detail

/// Record one resolved channel.  FIRST WINS: every resolver caches its answer in
/// a function-local static, so a second call is the same value, and refusing the
/// duplicate keeps the registry a set of channels rather than a call log.
inline void registerFormMask(const char* name, unsigned long long value, const char* source,
                             bool mined_sound) {
    std::lock_guard<std::mutex> lock(forms_detail::registryMutex());
    auto&                       records = forms_detail::registryCell();
    for (const FormMaskRecord& r : records)
        if (r.name == name) return;
    if (forms_detail::frozenCell())
        std::cerr << "[RASBERY][WARN][FORMS] " << name
                  << " resolved AFTER the case key was computed, so forms_digest does not "
                     "cover it.  The key under-describes this run's arithmetic; prime the "
                     "channel before the key if it can move a trajectory.\n";
    records.push_back(FormMaskRecord{name, value, source, mined_sound});
}

/// A snapshot of the registry in INSERTION order, for the receipts.
inline std::vector<FormMaskRecord> formMaskSnapshot() {
    std::lock_guard<std::mutex> lock(forms_detail::registryMutex());
    return forms_detail::registryCell();
}

/// The canonical bytes without latching -- for tests and receipts.
inline std::string formsPayload() {
    std::lock_guard<std::mutex> lock(forms_detail::registryMutex());
    return forms_detail::payloadLocked();
}

/// The canonical bytes, LATCHED on first call.  This is what the case key
/// digests, and latching is what makes two cases in one process key alike.
inline const std::string& formsPayloadFrozen() {
    static const std::string latched = [] {
        std::lock_guard<std::mutex> lock(forms_detail::registryMutex());
        forms_detail::frozenCell() = true;
        return forms_detail::payloadLocked();
    }();
    return latched;
}

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
    // The blanket pin, ranked BELOW the per-mask override above: this resolver
    // has no mined value, so `default` only renames the source it already took.
    // It is renamed anyway, because "this run was pinned" and "this run happened
    // to have no override" are different facts about a measurement.
    if (formsPin() == FormsPin::Default && source == std::string("build_default"))
        source = "pinned_default";
    std::cerr << "[RASBERY][FORMS] {\"mask\":\"" << mask_name << "\",\"value\":\"0x"
              << std::hex << value << std::dec << "\",\"source\":\"" << source
              << "\",\"build_default\":\"0x" << std::hex << build_default << std::dec
              << "\"}" << std::endl;
    registerFormMask(mask_name, value, source, /*mined_sound=*/false);
    return value;
}

/// Resolve one mask when the binary can MINE the host's answer for itself.
///
/// THE PRECEDENCE, AND WHY IT IS THIS WAY ROUND:
///
///   1. the environment override, when it parses -- a binary built on one host
///      and validated against a reference produced on another has to be able to
///      say so, and a human who typed the variable meant it;
///   2. the MINED value, when the mining is sound -- it is a measurement of THIS
///      binary on THIS machine, which is strictly better evidence than a
///      constant somebody measured on some other machine;
///   3. the build default, only when the mining could not reach zero mismatches,
///      and then LOUDLY, because at that point nothing here knows the contract.
///
/// The build default is therefore no longer a value the run depends on; it is
/// kept as the thing the receipt compares against, so a host whose contraction
/// differs from the shipped one says so in one line instead of being discovered
/// three campaigns later as an unexplained ULP.
inline unsigned long long resolveCalibratedFormMask(const char* env_name,
                                                    unsigned long long build_default,
                                                    unsigned long long mined,
                                                    bool mined_sound,
                                                    const char* mask_name) {
    unsigned long long value  = build_default;
    const char*        source = "build_default";

    if (mined_sound) {
        value  = mined;
        source = (mined == build_default) ? "mined_matches_default" : "mined";
        if (mined != build_default) {
            std::cerr << "[RASBERY][FORMS] " << mask_name << ": this host contracts 0x"
                      << std::hex << mined << " where the build default says 0x"
                      << build_default << std::dec
                      << "; using the mined value.  The default is a record of the"
                         " machine it was measured on, not of the physics.\n";
        }
    } else {
        std::cerr << "[RASBERY][WARN][FORMS] " << mask_name
                  << ": the contraction mining did not reach a bit-exact mask on this"
                     " host; falling back to the build default 0x"
                  << std::hex << build_default << std::dec
                  << ".  Device/host bit-equality is NOT established for this run.\n";
        source = "build_default_mining_failed";
    }

    // RASBERY_FORMS_PIN=default, ranked between the mining and the per-mask
    // override: it overrules the host's measurement (that is the whole point --
    // it buys the SHIPPED contraction on every host) and is itself overruled by
    // a per-channel override, because that was a specific request.
    if (formsPin() == FormsPin::Default) {
        value  = build_default;
        source = "pinned_default";
    }

    const char* raw = std::getenv(env_name);
    if (raw != nullptr) {
        unsigned long long parsed = 0;
        if (parseFormMask(raw, parsed)) {
            value  = parsed;
            source = "env";
        } else {
            std::cerr << "[RASBERY][WARN][FORMS] " << env_name << "=\"" << raw
                      << "\" is not a number; ignoring the override. "
                         "A malformed override must not silently pass for a valid one.\n";
        }
    }

    std::cerr << "[RASBERY][FORMS] {\"mask\":\"" << mask_name << "\",\"value\":\"0x"
              << std::hex << value << std::dec << "\",\"source\":\"" << source
              << "\",\"build_default\":\"0x" << std::hex << build_default << std::dec
              << "\",\"mined\":\"0x" << std::hex << mined << std::dec
              << "\",\"mined_sound\":" << (mined_sound ? 1 : 0) << "}" << std::endl;
    registerFormMask(mask_name, value, source, mined_sound);
    return value;
}

} // namespace rasbery::gpu
