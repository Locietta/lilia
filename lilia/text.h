#pragma once

#include <string>
#include <string_view>

#include <lighter/encoding/utf8.h>

#include <lighter/types.hpp>

namespace lilia {

inline std::string bounded_utf8(std::string_view text, lighter::types::usize maximum_bytes) {
    auto result = lighter::encoding::utf8::sanitize(text);
    if (result.size() <= maximum_bytes) return result;
    const auto complete = lighter::encoding::utf8::complete_prefix_len(std::string_view(result).substr(0, maximum_bytes));
    result.resize(complete);
    return result;
}

/// `text` without the ASCII spaces, tabs and line breaks around it.
inline std::string_view trim(std::string_view text) {
    constexpr std::string_view k_space = " \t\r\n";
    const auto first = text.find_first_not_of(k_space);
    if (first == std::string_view::npos) return {};
    return text.substr(first, text.find_last_not_of(k_space) - first + 1);
}

} // namespace lilia
