#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

namespace rasbery::cuda_transfer {

/// Host-side byte-exact shadow for a device upload.
///
/// `matches` deliberately uses memcmp rather than floating-point equality:
/// +0/-0, NaN payloads and every last mantissa bit are part of RASBERY's
/// reproducibility contract.  A caller must call commit() only after the
/// corresponding asynchronous H2D copy has completed successfully.  Until
/// then the previous shadow continues to describe the bytes known to reside
/// on the device.
template <class T>
class ByteExactMirror {
public:
    [[nodiscard]] bool valid() const noexcept { return _valid; }
    [[nodiscard]] std::size_t size() const noexcept {
        return _bytes.size() / sizeof(T);
    }

    [[nodiscard]] bool matches(const T* data, std::size_t count) const noexcept {
        if (!_valid || data == nullptr || count != size()) return false;
        return std::memcmp(_bytes.data(), data, count * sizeof(T)) == 0;
    }

    void commit(const T* data, std::size_t count) {
        if (data == nullptr) {
            invalidate();
            return;
        }
        _bytes.resize(count * sizeof(T));
        if (count != 0)
            std::memcpy(_bytes.data(), data, count * sizeof(T));
        _valid = true;
    }

    void invalidate() noexcept {
        _valid = false;
        _bytes.clear();
    }

private:
    std::vector<unsigned char> _bytes;
    bool                       _valid = false;
};

} // namespace rasbery::cuda_transfer
