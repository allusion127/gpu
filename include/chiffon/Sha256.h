#pragma once

// SHA-256, streaming, dependency-free.
//
// WHY IT IS ITS OWN HEADER.  It was written once, inside
// BatchLightResult::Sha256File, to digest the two provenance FILES a light
// receipt names.  WP10.1 needs the same digest over an in-memory canonical
// PAYLOAD (CaseKey.h), and a second copy of a hash function is the kind of
// duplication that ends with two answers to one question -- a cache key that
// disagrees with the receipt that named it.  So the transform moved here,
// unchanged, with an update/final interface, and both callers feed it.
//
// It is deliberately NOT a cryptographic claim about an adversary.  It is a
// content identity between runs of the same code base, and the reason it is
// SHA-256 rather than the FNV-1a the trajectory digest uses is that a
// duplicate-suppression cache addresses millions of payloads, where a 64-bit
// digest's birthday odds stop being negligible.

#include <array>
#include <cstdint>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace rasbery {

class Sha256 {
public:
    void update(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        _total += size;
        while (size > 0) {
            const std::size_t room = 64 - _fill;
            const std::size_t take = size < room ? size : room;
            for (std::size_t i = 0; i < take; ++i) _block[_fill + i] = bytes[i];
            _fill += take;
            bytes += take;
            size -= take;
            if (_fill == 64) {
                transform(_block.data());
                _fill = 0;
            }
        }
    }

    void update(std::string_view text) { update(text.data(), text.size()); }

    /// The digest as 64 lowercase hex characters.  Finalising twice is a
    /// programming error, not a supported operation.
    [[nodiscard]] std::string hex() {
        const std::uint64_t bits = _total * 8;
        std::uint8_t        pad  = 0x80;
        update(&pad, 1);
        pad = 0x00;
        while (_fill != 56) update(&pad, 1);
        std::array<std::uint8_t, 8> tail{};
        for (unsigned i = 0; i < 8; ++i)
            tail[i] = static_cast<std::uint8_t>(bits >> (8 * (7 - i)));
        // update() would add these eight bytes to _total, which no longer
        // matters: the length has already been captured above.
        for (unsigned i = 0; i < 8; ++i) _block[_fill + i] = tail[i];
        transform(_block.data());
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto word : _h) out << std::setw(8) << word;
        return out.str();
    }

    /// One-shot over a byte string.
    static std::string hexOf(std::string_view text) {
        Sha256 sha;
        sha.update(text);
        return sha.hex();
    }

private:
    static std::uint32_t rotr(std::uint32_t x, unsigned n) {
        return (x >> n) | (x << (32u - n));
    }

    void transform(const std::uint8_t* bytes) {
        static constexpr std::array<std::uint32_t, 64> K = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

        std::array<std::uint32_t, 64> w{};
        for (unsigned i = 0; i < 16; ++i)
            w[i] = (static_cast<std::uint32_t>(bytes[4 * i]) << 24) |
                   (static_cast<std::uint32_t>(bytes[4 * i + 1]) << 16) |
                   (static_cast<std::uint32_t>(bytes[4 * i + 2]) << 8) |
                   static_cast<std::uint32_t>(bytes[4 * i + 3]);
        for (unsigned i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = _h[0], b = _h[1], c = _h[2], d = _h[3];
        std::uint32_t e = _h[4], f = _h[5], g = _h[6], hh = _h[7];
        for (unsigned i = 0; i < 64; ++i) {
            const std::uint32_t s1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch  = (e & f) ^ ((~e) & g);
            const std::uint32_t t1  = hh + s1 + ch + K[i] + w[i];
            const std::uint32_t s0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2  = s0 + maj;
            hh = g;
            g  = f;
            f  = e;
            e  = d + t1;
            d  = c;
            c  = b;
            b  = a;
            a  = t1 + t2;
        }
        _h[0] += a;
        _h[1] += b;
        _h[2] += c;
        _h[3] += d;
        _h[4] += e;
        _h[5] += f;
        _h[6] += g;
        _h[7] += hh;
    }

    std::array<std::uint32_t, 8> _h = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::array<std::uint8_t, 64> _block{};
    std::size_t                  _fill  = 0;
    std::uint64_t                _total = 0;
};

} // namespace rasbery
