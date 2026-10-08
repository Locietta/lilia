#include "learner.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <format>
#include <span>
#include <unordered_map>
#include <utility>
#include <variant>

#include <glaze/json.hpp>

#include <xmake/bin2obj/learner_md.hpp>
#include <xmake/bin2obj/review_md.hpp>

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

/// Insults slip past the prompt now and then; lilia never attacks people, so
/// these are dropped whatever the model says. Insults whatever the target:
constexpr std::array<std::string_view, 14> k_slurs{"煞笔", "傻逼", "傻b",  "傻B",  "sb",   "SB",   "弱智",
                                                   "脑残", "智障", "去死", "nmsl", "你妈", "尼玛", "狗东西"};

/// Fine about things ("这卡有点区", "这道具纯纯垃圾"), not about people:
/// dropped only when the pattern is aimed at a person. (Not "杂鱼": that is
/// playful little-devil teasing.)
constexpr std::array<std::string_view, 11> k_put_downs{"区", "菜", "垃圾", "废物", "肺物", "笨蛋", "笨", "滚", "闭嘴", "艾草", "没用"};

/// A situation about someone (别人/对方/某人...) or a phrase said to "你".
constexpr std::array<std::string_view, 9> k_person_markers{"别人", "对方", "某人", "某个人", "群友", "他人", "人家", "朋友", "骂人"};

/// Situations that describe putting someone else down.
constexpr std::array<std::string_view, 3> k_blocked_situations{"骂人", "让对方", "让人闭嘴"};

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

/// A tool whose single argument is a list of numbers (as strings).
provider::ToolDefinition pick_tool(std::string name, std::string description, std::string field, std::string field_description) {
    glz::generic item_schema;
    std::ignore = glz::read_json(item_schema, R"({"type": "string"})");
    provider::SchemaProperty list{.type = "array", .description = std::move(field_description)};
    list.items.emplace(std::move(item_schema));
    std::string required = field;
    return {
        .name = std::move(name),
        .description = std::move(description),
        .input_schema = {.properties = {{std::move(field), std::move(list)}}, .required = {std::move(required)}},
    };
}

} // namespace

bool blocked_expression(std::string_view situation, std::string_view style) {
    const auto contains = [](std::string_view text, std::string_view word) { return text.find(word) != std::string_view::npos; };
    const auto in_either = [&](std::string_view word) { return contains(style, word) || contains(situation, word); };
    if (std::ranges::any_of(k_slurs, in_either)) return true;
    if (std::ranges::any_of(k_blocked_situations, [&](std::string_view word) { return contains(situation, word); })) return true;
    const bool at_person =
        contains(style, "你") || std::ranges::any_of(k_person_markers, [&](std::string_view word) { return contains(situation, word); });
    return at_person && std::ranges::any_of(k_put_downs, in_either);
}

std::string normalize_style(std::string_view style) {
    std::string out;
    for (usize i = 0; i < style.size();) {
        bool replaced = false;
        for (const std::string_view quote : {"「", "」", "“", "”", "『", "』"}) {
            if (style.substr(i).starts_with(quote)) {
                out += '"';
                i += quote.size();
                replaced = true;
                break;
            }
        }
        if (!replaced) out += style[i++];
    }
    // 说"…" / 回"…" / 用"…" mean the same thing; one form lets repeats count.
    for (const std::string_view verb : {"说\"", "回\""}) {
        if (out.starts_with(verb)) out = "用\"" + out.substr(verb.size());
    }
    return out;
}

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
            auto style = normalize_style(text_field(entry, "style"));
            const auto source_text = text_field(entry, "source");
            usize source = 0;
            const auto [end, ec] = std::from_chars(source_text.data(), source_text.data() + source_text.size(), source);
            if (ec != std::errc{} || source == 0 || source > line_count) continue;
            if (!fits(situation) || !fits(style) || blocked_expression(situation, style)) continue;
            learned.push_back({.situation = std::move(situation), .style = std::move(style)});
            if (learned.size() == 5) break;
        }
        break;
    }
    return learned;
}

LearnerRequest build_review_request(std::string_view prompt, const std::vector<Learned> &learned) {
    LearnerRequest request;
    provider::append_system(request.history, std::string(prompt));
    std::string list;
    for (usize i = 0; i < learned.size(); ++i) list += std::format("{}. 当{}时，可以{}\n", i + 1, learned[i].situation, learned[i].style);
    provider::append_user(request.history, std::move(list));
    request.tools.push_back(
        pick_tool("review_expressions", "提交不适合留下的说法编号", "reject", "不适合留下的编号，例如 [\"2\"]，都合适就为空"));
    return request;
}

LearnerRequest build_selection_request(std::string_view prompt, const SelectionInput &input) {
    LearnerRequest request;
    provider::append_system(request.history, std::string(prompt));
    std::string user = "最近的聊天：\n";
    for (const auto &line : input.recent) user += line + "\n";
    user += std::format("\n最新的一条：{}\n", input.target);
    if (!input.intent.empty()) user += std::format("她想表达的意思：{}\n", input.intent);
    user += "\n学来的说法：\n";
    for (usize i = 0; i < input.candidates.size(); ++i) user += std::format("{}. {}\n", i + 1, input.candidates[i]);
    provider::append_user(request.history, std::move(user));
    request.tools.push_back(pick_tool("select_expressions", "提交这次可以参考的说法编号", "ids", "选中的编号，0 到 3 个，可以为空"));
    return request;
}

std::vector<usize> parse_picks(const std::vector<provider::OutputItem> &items, std::string_view tool, std::string_view field_name,
                               usize count, usize limit) {
    std::vector<usize> picks;
    for (const auto &item : items) {
        const auto *call = std::get_if<provider::ToolCallItem>(&item);
        if (!call || call->call.name != tool) continue;
        const auto *list = field(call->call.input, field_name);
        if (!list || !list->is_array()) break;
        for (const auto &entry : list->get_array()) {
            usize number = 0;
            if (entry.is_number()) {
                number = static_cast<usize>(entry.get_number());
            } else if (entry.is_string()) {
                std::string_view text = trim(entry.get_string());
                if (text.starts_with('#')) text.remove_prefix(1);
                if (std::from_chars(text.data(), text.data() + text.size(), number).ec != std::errc{}) continue;
            } else {
                continue;
            }
            if (number >= 1 && number <= count && !std::ranges::contains(picks, number - 1)) picks.push_back(number - 1);
            if (picks.size() >= limit) break;
        }
        break;
    }
    return picks;
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
      review_prompt(
          chat::fill_template({reinterpret_cast<const char *>(xmake::review_md.data()), xmake::review_md.size()}, {{"name", name}})),
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

namespace {

/// One learning-side model call, recorded as chat type "learn".
Task<std::optional<std::vector<provider::OutputItem>>> run_call(Learner *learner, i64 group, const LearnerRequest &request,
                                                                std::string_view purpose) {
    std::vector<provider::OutputItem> items;
    const provider::StreamCallbacks callbacks{
        .on_item_completed = [&items](const provider::OutputItem &item) { items.push_back(item); },
    };
    const i64 started = learner->now();
    auto completion = co_await learner->provider->complete(request.history, request.tools, callbacks);

    store::LlmCall call{.time = started, .chat = {.type = "learn", .id = group}, .model = "learner"};
    if (completion.has_error()) {
        call.error = completion.error().message();
        log::warn("{} call failed: {}", purpose, call.error);
    } else {
        if (!completion->model.empty()) call.model = completion->model;
        call.input_tokens = completion->usage.input_tokens;
        call.output_tokens = completion->usage.output_tokens;
        call.cache_read_tokens = completion->usage.cache_read_tokens;
        call.stop = completion->stop_detail;
        call.request_id = completion->request_id;
    }
    if (auto recorded = learner->store.add_llm_call(call); recorded.has_error())
        log::error("cannot record model call: {}", recorded.error().message());
    if (completion.has_error()) co_return std::nullopt;
    co_return std::move(items);
}

} // namespace

Task<std::optional<std::vector<usize>>> Learner::review(i64 group, const std::vector<Learned> &learned) {
    if (learned.empty()) co_return std::vector<usize>{};
    const auto items = co_await run_call(this, group, build_review_request(review_prompt, learned), "review");
    if (!items) co_return std::nullopt;
    // Asking what to drop (not what to keep) makes the reviewer less picky.
    const auto rejected = parse_picks(*items, "review_expressions", "reject", learned.size(), learned.size());
    if (std::ranges::none_of(*items, [](const provider::OutputItem &item) {
            const auto *call = std::get_if<provider::ToolCallItem>(&item);
            return call && call->call.name == "review_expressions";
        })) {
        co_return std::nullopt;
    }
    std::vector<usize> kept;
    for (usize i = 0; i < learned.size(); ++i) {
        if (!std::ranges::contains(rejected, i)) kept.push_back(i);
    }
    co_return kept;
}

Task<std::vector<Learned>> Learner::learn(i64 group, std::vector<Line> lines) {
    const i64 started = now();
    const auto items = co_await run_call(this, group, build_learner_request(prompt, lines), "learning");
    if (!items) co_return std::vector<Learned>{};
    const auto extracted = parse_learned(*items, lines.size());

    // Nothing is stored unless the review passes it; a failed review keeps nothing.
    const auto kept = co_await review(group, extracted);
    std::vector<Learned> learned;
    if (kept) {
        for (const auto index : *kept) learned.push_back(extracted[index]);
    }
    for (const auto &pattern : learned) {
        if (auto added = store.learn_expression(pattern.situation, pattern.style, started); added.has_error()) {
            log::error("cannot store expression: {}", added.error().message());
        }
    }
    if (config.max_expressions > 0) {
        if (auto pruned = store.prune_expressions(config.max_expressions); pruned.has_error())
            log::error("cannot prune expressions: {}", pruned.error().message());
    }
    log::info("learned {} expression(s) from {} lines ({} extracted, {} rejected by review)", learned.size(), lines.size(),
              extracted.size(), extracted.size() - learned.size());
    co_return learned;
}

} // namespace lilia::learn
