#include "learner.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <format>
#include <span>
#include <unordered_map>
#include <utility>
#include <variant>

#include <glaze/json.hpp>

#include <xmake/bin2obj/learner_md.hpp>

#include <lilia/chat/context.h>
#include <lilia/chat/delivery.h>
#include <lilia/log.h>
#include <lilia/onebot/message.h>

namespace lilia::learn {

using lighter::Task;

namespace {

constexpr usize k_max_field_code_points = 30;
constexpr usize k_max_line_code_points = 150;

std::string_view trim(std::string_view text) {
    constexpr std::string_view k_space = " \t\r\n";
    const auto first = text.find_first_not_of(k_space);
    if (first == std::string_view::npos) return {};
    const auto last = text.find_last_not_of(k_space);
    return text.substr(first, last - first + 1);
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
    if (value->is_string()) return std::string(trim(value->get_string()));
    if (value->is_number()) return std::to_string(static_cast<i64>(value->get_number()));
    return {};
}

/// "被问具体内容时" -> "被问具体内容"; the template adds "当…时".
std::string strip_time_suffix(std::string text) {
    for (const std::string_view suffix : {"的时候", "时候", "时"}) {
        if (text.ends_with(suffix) && text.size() > suffix.size()) {
            text.resize(text.size() - suffix.size());
            break;
        }
    }
    return text;
}

bool fits(std::string_view text) { return !text.empty() && chat::count_code_points(text) <= k_max_field_code_points; }

/// Spawned tasks return nothing; the learned patterns are already stored.
Task<> learn_batch(Learner *learner, i64 group, std::vector<Line> lines) { std::ignore = co_await learner->learn(group, std::move(lines)); }

provider::ToolDefinition record_tool() {
    glz::generic item_schema;
    std::ignore = glz::read_json(item_schema, R"({"type": "object", "properties": {
        "situation": {"type": "string", "description": "场景，不超过 20 个字"},
        "style": {"type": "string", "description": "说法、句式或梗，不超过 20 个字"},
        "source": {"type": "string", "description": "来源行的编号"}},
        "required": ["situation", "style", "source"]})");
    provider::SchemaProperty items{.type = "array", .description = "总结结果，可以为空"};
    items.items.emplace(std::move(item_schema));
    return {
        .name = "record_expressions",
        .description = "提交总结出的 0 到 5 条说话方式",
        .input_schema = {.properties = {{"items", std::move(items)}}, .required = {"items"}},
    };
}

} // namespace

LearnerRequest build_learner_request(std::string_view prompt, const std::vector<Line> &lines) {
    LearnerRequest request;
    provider::append_system(request.history, std::string(prompt));
    std::unordered_map<i64, usize> speakers;
    std::string log;
    for (usize i = 0; i < lines.size(); ++i) {
        const auto [it, inserted] = speakers.try_emplace(lines[i].sender, speakers.size() + 1);
        log += std::format("{}. 群友{}: {}\n", i + 1, it->second, lines[i].text);
    }
    provider::append_user(request.history, std::move(log));
    request.tools.push_back(record_tool());
    return request;
}

std::vector<Learned> parse_learned(const std::vector<provider::OutputItem> &items, usize line_count) {
    std::vector<Learned> learned;
    for (const auto &item : items) {
        const auto *tool = std::get_if<provider::ToolCallItem>(&item);
        if (!tool || tool->call.name != "record_expressions") continue;
        const auto *list = field(tool->call.input, "items");
        if (!list || !list->is_array()) break;
        for (const auto &entry : list->get_array()) {
            auto situation = strip_time_suffix(text_field(entry, "situation"));
            auto style = text_field(entry, "style");
            const auto source_text = text_field(entry, "source");
            usize source = 0;
            const auto [end, ec] = std::from_chars(source_text.data(), source_text.data() + source_text.size(), source);
            if (ec != std::errc{} || source == 0 || source > line_count) continue;
            if (!fits(situation) || !fits(style)) continue;
            learned.push_back({.situation = std::move(situation), .style = std::move(style)});
            if (learned.size() == 5) break;
        }
        break;
    }
    return learned;
}

std::string learnable_text(const onebot::Message &message) {
    std::string text;
    for (const auto &segment : message) {
        if (segment.type == "text") text += onebot::data_string(segment, "text").value_or("");
    }
    auto trimmed = std::string(trim(text));
    if (trimmed.starts_with('/') || chat::count_code_points(trimmed) > k_max_line_code_points) return {};
    return trimmed;
}

Learner::Learner(const LearningConfig &config, std::string name, store::Store &store, provider::Provider provider, Spawn spawn)
    : now([] { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }),
      config(config),
      prompt(chat::fill_template({reinterpret_cast<const char *>(xmake::learner_md.data()), xmake::learner_md.size()}, {{"name", name}})),
      store(store), provider(std::move(provider)), spawn(std::move(spawn)) {}

void Learner::observe(const onebot::MessageEvent &message, i64 self_id) {
    if (!config.enabled || !message.is_group() || message.user_id == self_id) return;
    if (std::ranges::contains(config.exclude_groups, *message.group_id)) return;
    auto text = learnable_text(message.message);
    if (text.empty()) return;
    auto &buffer = buffers[*message.group_id];
    buffer.push_back({.sender = message.user_id, .text = std::move(text)});
    if (buffer.size() < std::max<usize>(config.batch_messages, 2)) return;
    spawn(learn_batch(this, *message.group_id, std::exchange(buffer, {})));
}

Task<std::vector<Learned>> Learner::learn(i64 group, std::vector<Line> lines) {
    const auto request = build_learner_request(prompt, lines);
    std::vector<provider::OutputItem> items;
    const provider::StreamCallbacks callbacks{
        .on_item_completed = [&items](const provider::OutputItem &item) { items.push_back(item); },
    };
    const i64 started = now();
    auto completion = co_await provider->complete(request.history, request.tools, callbacks);

    store::LlmCall call{.time = started, .chat = {.type = "learn", .id = group}, .model = "learner"};
    if (completion.has_error()) {
        call.error = completion.error().message();
        log::warn("learning call failed: {}", call.error);
    } else {
        if (!completion->model.empty()) call.model = completion->model;
        call.input_tokens = completion->usage.input_tokens;
        call.output_tokens = completion->usage.output_tokens;
        call.cache_read_tokens = completion->usage.cache_read_tokens;
        call.stop = completion->stop_detail;
        call.request_id = completion->request_id;
    }
    if (auto recorded = store.add_llm_call(call); recorded.has_error())
        log::error("cannot record model call: {}", recorded.error().message());
    if (completion.has_error()) co_return std::vector<Learned>{};

    auto learned = parse_learned(items, lines.size());
    for (const auto &pattern : learned) {
        if (auto added = store.learn_expression(pattern.situation, pattern.style, started); added.has_error()) {
            log::error("cannot store expression: {}", added.error().message());
        }
    }
    if (auto pruned = store.prune_expressions(config.max_expressions); pruned.has_error()) {
        log::error("cannot prune expressions: {}", pruned.error().message());
    }
    log::info("learned {} expression(s) from {} lines", learned.size(), lines.size());
    co_return std::move(learned);
}

} // namespace lilia::learn
