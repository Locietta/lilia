#include <cstdio>
#include <string>
#include <string_view>

#include <glaze/json.hpp>

#include <lilia/provider/detail/anthropic_protocol.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;

const glz::generic *field(const glz::generic &json, std::string_view key) {
    if (!json.is_object()) return nullptr;
    const auto &object = json.get_object();
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

bool cached(const glz::generic &json) { return field(json, "cache_control") != nullptr; }

glz::generic encode(const provider::History &history, bool prompt_cache) {
    const std::vector<provider::ToolDefinition> tools{{.name = "first"}, {.name = "last"}};
    auto body = anthropic::protocol::encode_complete_request(history, tools, {.model = "m", .prompt_cache = prompt_cache});
    require(body.has_value(), "request encodes");
    glz::generic json;
    require(!glz::read_json(json, *body), "request is JSON");
    return json;
}

void test_system_cache() {
    provider::History history;
    provider::append_system(history, "PERSONA");
    provider::append_developer(history, "NOW");
    provider::append_user(history, "hi");

    // Without caching (DeepSeek, MiniMax) system stays one string, no markers.
    const auto plain = encode(history, false);
    const auto *prompt = field(plain, "system");
    require(prompt && prompt->is_string(), "plain system is a string");
    const std::string expected = "Instruction hierarchy: SYSTEM instructions take precedence over DEVELOPER instructions.\n"
                                 "\n[SYSTEM]\nPERSONA\n[DEVELOPER]\nNOW";
    require(prompt->get_string() == expected, "plain system text");
    require(!cached(plain), "no top-level cache marker");
    for (const auto &tool : field(plain, "tools")->get_array()) require(!cached(tool), "no tool cache marker");

    // With caching the same text goes as blocks, the breakpoint after SYSTEM.
    const auto json = encode(history, true);
    const auto *blocks = field(json, "system");
    require(blocks && blocks->is_array() && blocks->get_array().size() == 2, "one system block per instruction");
    std::string joined;
    for (const auto &block : blocks->get_array()) {
        require(field(block, "type")->get_string() == "text", "text blocks");
        joined += field(block, "text")->get_string();
    }
    require(joined == expected, "blocks spell the same system text");
    const auto &stable = blocks->get_array()[0];
    require(cached(stable) && field(*field(stable, "cache_control"), "type")->get_string() == "ephemeral",
            "breakpoint on the SYSTEM block");
    require(!cached(blocks->get_array()[1]), "none on the DEVELOPER block");
    require(cached(json), "top-level automatic caching");
    const auto &tools = field(json, "tools")->get_array();
    require(!cached(tools[0]) && cached(tools[1]), "breakpoint on the last tool only");

    // A prompt that opens with DEVELOPER text has no stable part to mark.
    provider::History dynamic;
    provider::append_developer(dynamic, "NOW");
    provider::append_system(dynamic, "PERSONA");
    provider::append_user(dynamic, "hi");
    for (const auto &block : field(encode(dynamic, true), "system")->get_array()) {
        require(!cached(block), "no breakpoint after changing text");
    }
}

} // namespace

int main() {
    try {
        test_system_cache();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
