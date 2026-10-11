#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>

#include <lighter/types.hpp>

namespace lilia::net {

using namespace lighter::types;

/// Standard base64 (RFC 4648 section 4) with padding.
std::string base64_encode(std::string_view bytes);

/// Decodes standard base64; whitespace is skipped, padding is optional.
/// nullopt on any other character outside the alphabet.
std::optional<std::string> base64_decode(std::string_view text);

/// SHA-1 digest. Only for protocol checks such as the WebSocket accept key
/// and content addressing; never for anything security sensitive.
std::array<u8, 20> sha1(std::string_view bytes);

/// The same digest fed in pieces, so a large file can be hashed a chunk at
/// a time.
struct Sha1 {
    void update(std::string_view bytes);
    std::array<u8, 20> finish();

    std::array<u32, 5> state{0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    std::string pending; ///< fewer than 64 bytes not yet hashed
    u64 length = 0;      ///< bytes fed so far
};

} // namespace lilia::net
