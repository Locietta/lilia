#pragma once

#include <string>
#include <string_view>

#include <glaze/json.hpp>

#include <lilia/text.h>

namespace lilia {

// Reading the arguments of a model's tool call (glz::generic objects).

/// `object[key]`; null when `object` is not an object or lacks the key.
inline const glz::generic *field(const glz::generic &object, std::string_view key) {
    if (!object.is_object()) return nullptr;
    const auto &map = object.get_object();
    const auto it = map.find(key);
    return it == map.end() ? nullptr : &it->second;
}

/// A string field as it is; empty when absent or not a string.
inline std::string string_field(const glz::generic &object, std::string_view key) {
    const auto *value = field(object, key);
    return value && value->is_string() ? value->get_string() : std::string();
}

/// A string field trimmed, or a number written out; empty otherwise.
inline std::string text_field(const glz::generic &object, std::string_view key) {
    const auto *value = field(object, key);
    if (!value) return {};
    if (value->is_string()) return std::string(trim(value->get_string()));
    if (value->is_number()) return std::to_string(static_cast<long long>(value->get_number()));
    return {};
}

/// true, "true" or "是".
inline bool bool_field(const glz::generic &object, std::string_view key) {
    const auto *value = field(object, key);
    if (!value) return false;
    if (value->is_boolean()) return value->get_boolean();
    return value->is_string() && (value->get_string() == "true" || value->get_string() == "是");
}

} // namespace lilia
