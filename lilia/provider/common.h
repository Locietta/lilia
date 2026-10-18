#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <glaze/json.hpp>

namespace lilia::provider {

// Provider-neutral agent vocabulary only. Concrete clients intentionally keep
// their own history and wire types so provider-specific APIs remain available.

struct ToolCall {
    std::string id;
    std::string name;
    glz::generic input;
};

struct SchemaProperty {
    std::string type;
    std::string description;
    /// Full JSON Schema for array elements, including nested objects.
    std::optional<glz::generic> items;
};

struct InputSchema {
    std::string type = "object";
    std::map<std::string, SchemaProperty> properties;
    std::vector<std::string> required;
    bool additional_properties = false;
};

enum struct ToolKind {
    FUNCTION,
    WEB_SEARCH,
    WEB_FETCH,
};

struct ToolDefinition {
    ToolKind kind = ToolKind::FUNCTION;
    std::string name;
    std::string description;
    InputSchema input_schema;
};

} // namespace lilia::provider

template <>
struct glz::meta<lilia::provider::InputSchema> {
    using T = lilia::provider::InputSchema;
    static constexpr auto value =
        object("type", &T::type, "properties", &T::properties, "required", &T::required, "additionalProperties", &T::additional_properties);
};
