#include "memory.h"

#include <algorithm>
#include <charconv>
#include <format>

#include <glaze/json.hpp>

#include <lilia/chat/delivery.h>
#include <lilia/log.h>

namespace lilia::memory {

using lighter::Task;

namespace {

constexpr usize k_max_facts = 5;
constexpr usize k_min_fact_code_points = 4;
constexpr usize k_max_fact_code_points = 40;

std::string without_spaces(std::string_view text) {
    std::string out;
    for (const char c : text) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') out.push_back(c);
    }
    return out;
}

const glz::generic *field(const glz::generic &object, std::string_view key) {
    if (!object.is_object()) return nullptr;
    const auto &map = object.get_object();
    const auto it = map.find(key);
    return it == map.end() ? nullptr : &it->second;
}

std::string text_field(const glz::generic &object, std::string_view key) {
    const auto *value = field(object, key);
    if (!value) return {};
    if (value->is_string()) return value->get_string();
    if (value->is_number()) return std::to_string(static_cast<i64>(value->get_number()));
    return {};
}

/// "TA 在成都。" -> "TA在成都": one spelling, so repeats are caught.
std::string tidy_fact(std::string fact) {
    if (fact.starts_with("TA ")) fact.erase(2, 1);
    for (const std::string_view mark : {"。", "！", ".", "!"}) {
        while (fact.ends_with(mark)) fact.resize(fact.size() - mark.size());
    }
    return fact;
}

provider::ToolDefinition record_facts_tool() {
    glz::generic item_schema;
    std::ignore = glz::read_json(item_schema, R"({"type": "object", "properties": {
        "fact": {"type": "string", "description": "关于 TA 的事实，简短的中文陈述句"},
        "source": {"type": "string", "description": "来自 TA 的哪条发言（编号）"},
        "quote": {"type": "string", "description": "一字不差摘自那条发言的原话片段"}},
        "required": ["fact", "source", "quote"]})");
    provider::SchemaProperty list{.type = "array", .description = "提取出的事实，可以为空"};
    list.items.emplace(std::move(item_schema));
    return {
        .name = "record_facts",
        .description = "提交关于 TA 的稳定事实",
        .input_schema = {.properties = {{"facts", std::move(list)}}, .required = {"facts"}},
    };
}

std::string speaker(const store::StoredMessage &message, std::string_view self_name) {
    if (message.from_self) return std::string(self_name);
    return message.sender_name.empty() ? std::to_string(message.sender_id) : message.sender_name;
}

} // namespace

FactRequest build_fact_request(std::string_view prompt, std::string_view person, const std::vector<store::StoredMessage> &own,
                               const std::vector<store::StoredMessage> &context, std::string_view self_name) {
    FactRequest request;
    provider::append_system(request.history, std::string(prompt));
    std::string user = std::format("目标人物（TA）：{}\n\nTA 自己的发言：\n", person);
    for (usize i = 0; i < own.size(); ++i) user += std::format("{}. {}\n", i + 1, own[i].text);
    user += "\n附近的聊天（只用来理解上下文）：\n";
    for (const auto &message : context) user += std::format("{}: {}\n", speaker(message, self_name), message.text);
    provider::append_user(request.history, std::move(user));
    request.tools.push_back(record_facts_tool());
    return request;
}

std::vector<ExtractedFact> parse_facts(const std::vector<provider::OutputItem> &items, const std::vector<store::StoredMessage> &own) {
    std::vector<ExtractedFact> facts;
    for (const auto &item : items) {
        const auto *call = std::get_if<provider::ToolCallItem>(&item);
        if (!call || call->call.name != "record_facts") continue;
        const auto *list = field(call->call.input, "facts");
        if (!list || !list->is_array()) break;
        for (const auto &entry : list->get_array()) {
            auto fact = tidy_fact(text_field(entry, "fact"));
            auto quote = text_field(entry, "quote");
            std::string_view source_text = text_field(entry, "source");
            if (source_text.starts_with('#')) source_text.remove_prefix(1);
            usize source = 0;
            if (std::from_chars(source_text.data(), source_text.data() + source_text.size(), source).ec != std::errc{}) continue;
            if (source == 0 || source > own.size()) continue;
            const auto length = chat::count_code_points(fact);
            if (length < k_min_fact_code_points || length > k_max_fact_code_points) continue;
            // The quote must really be what TA wrote in that message.
            const auto needle = without_spaces(quote);
            if (needle.empty() || without_spaces(own[source - 1].text).find(needle) == std::string::npos) continue;
            facts.push_back({.fact = std::move(fact), .quote = std::move(quote), .source = source - 1});
            if (facts.size() == k_max_facts) break;
        }
        break;
    }
    return facts;
}

std::string render_facts(const std::vector<store::PersonFact> &facts) {
    std::string out;
    for (const auto &fact : facts) {
        std::string_view text = fact.fact;
        if (text.starts_with("TA")) text.remove_prefix(2);
        if (!out.empty()) out += "；";
        out += text;
    }
    return out;
}

Task<bool> Memory::remember(store::PersonFact fact) {
    auto existing = store.facts(fact.person_id, 0);
    if (existing.has_error()) {
        log::error("memory: {}", existing.error().message());
        co_return false;
    }
    if (std::ranges::any_of(*existing, [&](const store::PersonFact &known) { return known.fact == fact.fact; })) co_return false;
    auto vectors = co_await embed({fact.fact});
    if (vectors.has_error() || vectors->empty()) {
        log::warn("memory: cannot embed a fact: {}", vectors.has_error() ? vectors.error().message() : "no vector");
        co_return false;
    }
    const auto &vector = vectors->front();
    for (const auto &known : *existing) {
        if (known.model == config.embedding_model && cosine(decode_vector(known.vector), vector) >= config.duplicate_fact_score) {
            co_return false;
        }
    }
    fact.model = config.embedding_model;
    fact.vector = encode_vector(vector);
    if (auto added = store.add_fact(fact); added.has_error()) {
        log::error("memory: {}", added.error().message());
        co_return false;
    }
    log::info("memory: learned about {}: {}", fact.person_id, fact.fact);
    co_return true;
}

std::string Memory::people_notes(const std::vector<std::pair<i64, std::string>> &people) {
    std::string notes;
    for (const auto &[id, name] : people) {
        auto facts = store.facts(id, config.facts_per_person);
        if (facts.has_error() || facts->empty()) continue;
        notes += std::format("- {}：{}\n", name, render_facts(*facts));
    }
    return notes;
}

} // namespace lilia::memory
