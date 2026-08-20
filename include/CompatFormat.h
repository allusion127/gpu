#pragma once

// Prefer the standard implementation when the library actually provides it.
// Fall back to fmt only for older libstdc++ releases; selecting fmt merely
// because its header is installed makes both std::format overload sets visible
// on GCC 13+ and causes ambiguous calls.
#if __has_include(<version>)
#include <version>
#endif

#if defined(__cpp_lib_format)
#include <format>
#elif __has_include(<fmt/format.h>)
#define FMT_HEADER_ONLY
#include <fmt/format.h>
namespace std {
using fmt::format;
}
#else
#error "RASBERY requires either C++20 std::format or fmt/format.h"
#endif
