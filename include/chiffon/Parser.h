#pragma once

// Low-level text and HDF5 loading helpers used by the importer.

#include "milk.h"
#include "highfive/highfive.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace Chiffon {
namespace Parser {

// Lightweight zero-copy text parser for memory-mapped HGC file blocks.
// Holds two pointers (current position `p`, buffer `end`) into a buffer it does
// not own, and advances them in place. The stream-style operator>> overloads
// mimic std::istream without any allocation or copying of the source bytes.
class FastParser {
public:
    const char* p;
    const char* end;

    FastParser(const char* start, size_t len)
        : p(start), end(start + len) {
    }

    void skipWhitespace() {
        while (p < end && std::isspace(static_cast<unsigned char>(*p)))
            ++p;
    }

    // Extract the next whitespace-delimited token as a string.
    FastParser& operator>>(std::string& out) {
        skipWhitespace();
        if (p >= end) return *this;
        const char* start = p;
        while (p < end && !std::isspace(static_cast<unsigned char>(*p)))
            ++p;
        out.assign(start, p - start);
        return *this;
    }

    // Parse the next token as a double (handles scientific notation via strtod).
    FastParser& operator>>(double& out) {
        skipWhitespace();
        if (p >= end) return *this;
        char* next;
        out = std::strtod(p, &next);
        p   = next;
        return *this;
    }

    // Parse the next token as a base-10 int.
    FastParser& operator>>(int& out) {
        skipWhitespace();
        if (p >= end) return *this;
        char* next;
        out = static_cast<int>(std::strtol(p, &next, 10));
        p   = next;
        return *this;
    }

    // Read up to (and consume) the next line terminator into `out`.
    bool getline(std::string& out) {
        if (p >= end) return false;
        const char* start = p;
        while (p < end && *p != '\n' && *p != '\r')
            ++p;
        out.assign(start, p - start);
        if (p < end && *p == '\r') ++p;
        if (p < end && *p == '\n') ++p;
        return true;
    }

    // Advance past one whitespace-delimited token without storing it.
    void skipToken() {
        skipWhitespace();
        if (p >= end) return;
        while (p < end && !std::isspace(static_cast<unsigned char>(*p)))
            ++p;
    }

    operator bool() const { return p < end; }
};

// Copy one contiguous slice and advance the shared offset.
inline void LoadVector(milk::Vector<double>& dst, const std::vector<double>& src,
                       size_t& offset, size_t count) {
    if (offset > src.size() || count > src.size() - offset)
        throw std::runtime_error("Truncated flat HDF5 vector.");
    dst.assign(count, 0.0);
    if (count > 0)
        std::copy_n(src.data() + offset, count, dst.data());
    offset += count;
}

} // namespace Parser
} // namespace Chiffon
