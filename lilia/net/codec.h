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

/// SHA-1 digest. Only for protocol checks such as the WebSocket accept key;
/// never for anything security sensitive.
std::array<u8, 20> sha1(std::string_view bytes);

} // namespace lilia::net
