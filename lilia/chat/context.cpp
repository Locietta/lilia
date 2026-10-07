#include "context.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <format>
#include <unordered_set>
#include <variant>

#include <lilia/text.h>

namespace lilia::chat {

namespace {

std::string_view trim(std::string_view text) {
    constexpr std::string_view k_space = " \t\r\n";
    const auto first = text.find_first_not_of(k_space);
    if (first == std::string_view::npos) return {};
    const auto last = text.find_last_not_of(k_space);
    return text.substr(first, last - first + 1);
}

std::chrono::sys_seconds local_seconds(i64 unix_seconds, i32 utc_offset_minutes) {
    return std::chrono::sys_seconds(std::chrono::seconds(unix_seconds + i64{utc_offset_minutes} * 60));
}

std::string offset_label(i32 utc_offset_minutes) {
    const char sign = utc_offset_minutes < 0 ? '-' : '+';
    const i32 magnitude = utc_offset_minutes < 0 ? -utc_offset_minutes : utc_offset_minutes;
    if (magnitude % 60 == 0) return std::format("UTC{}{}", sign, magnitude / 60);
    return std::format("UTC{}{}:{:02}", sign, magnitude / 60, magnitude % 60);
}

std::string display_name(const store::StoredMessage &message) {
    return message.sender_name.empty() ? std::to_string(message.sender_id) : message.sender_name;
}

/// Shared "where and when" facts for both calls.
std::string chat_facts(const ChatInfo &info, i64 now, i32 utc_offset_minutes) {
    std::string facts = std::format("现在是 {}（{}）。", format_local_time(now, utc_offset_minutes), offset_label(utc_offset_minutes));
    if (info.chat.type == "group") {
        facts += std::format("这是 QQ 群「{}」。", info.chat_name);
    } else {
        facts += std::format("这是和「{}」的私聊。", info.chat_name);
    }
    facts += std::format("{}的 QQ 昵称是「{}」，别人 @ 她时会显示成「@{}」。", info.name, info.self_name, info.self_name);
    return facts;
}

/// Newest-first walk that keeps whole lines within `budget` bytes, returned
/// oldest first.
template <typename Render>
std::vector<usize> fit_lines(const std::vector<store::StoredMessage> &recent, usize budget, Render &&render) {
    std::vector<usize> kept;
    usize used = 0;
    for (usize i = recent.size(); i-- > 0;) {
        const auto size = render(recent[i]).size() + 1;
        if (used + size > budget && !kept.empty()) break;
        used += size;
        kept.push_back(i);
    }
    std::ranges::reverse(kept);
    return kept;
}

provider::SchemaProperty property(std::string type, std::string description) {
    return {.type = std::move(type), .description = std::move(description)};
}

const glz::generic *field(const glz::generic &input, std::string_view key) {
    if (!input.is_object()) return nullptr;
    const auto &object = input.get_object();
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

std::string string_field(const glz::generic &input, std::string_view key) {
    const auto *value = field(input, key);
    return value && value->is_string() ? value->get_string() : std::string();
}

bool bool_field(const glz::generic &input, std::string_view key) {
    const auto *value = field(input, key);
    if (!value) return false;
    if (value->is_boolean()) return value->get_boolean();
    if (value->is_string()) return value->get_string() == "true";
    return false;
}

/// "#12", "12" or "#12 小明: ..." -> 12.
std::optional<usize> parse_target(std::string_view text) {
    text = trim(text);
    if (text.starts_with('#')) text.remove_prefix(1);
    usize value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end == text.data()) return std::nullopt;
    return value;
}

std::string_view style_instruction(ReplyStyle style) {
    switch (style) {
        case ReplyStyle::SHORT: return "这次就一句话，十几个字以内，一两个词也行。";
        case ReplyStyle::NORMAL: return "这次一两句话，加起来四十个字以内。";
        case ReplyStyle::LONG: return "这次可以多说几句把事情讲清楚，但还是聊天的口气，一百个字以内。";
    }
    return "";
}

/// Removes "[HH:MM] " and "name(qq): " prefixes the model may imitate.
std::string_view strip_speaker_label(std::string_view text, const ChatInfo &info, const std::vector<std::string> &aliases) {
    text = trim(text);
    if (text.starts_with('[')) {
        const auto close = text.find(']');
        if (close != std::string_view::npos && close <= 20) text = trim(text.substr(close + 1));
    }
    std::vector<std::string_view> names{info.self_name, info.name};
    for (const auto &alias : aliases) names.push_back(alias);
    for (const auto name : names) {
        if (name.empty() || !text.starts_with(name)) continue;
        auto rest = text.substr(name.size());
        if (rest.starts_with('(') || rest.starts_with("（")) {
            const auto close = rest.find_first_of(")）");
            if (close == std::string_view::npos || close > 32) continue;
            rest = rest.substr(rest[close] == ')' ? close + 1 : close + std::string_view("）").size());
        }
        if (rest.starts_with(':')) return trim(rest.substr(1));
        if (rest.starts_with("：")) return trim(rest.substr(std::string_view("：").size()));
    }
    return text;
}

} // namespace

std::string format_local_time(i64 unix_seconds, i32 utc_offset_minutes) {
    return std::format("{:%Y-%m-%d %H:%M}", local_seconds(unix_seconds, utc_offset_minutes));
}

i64 local_day_start(i64 unix_seconds, i32 utc_offset_minutes) {
    const i64 offset = i64{utc_offset_minutes} * 60;
    const i64 local = unix_seconds + offset;
    const i64 day = local >= 0 ? local / 86'400 : (local - 86'399) / 86'400;
    return day * 86'400 - offset;
}

std::string render_line(const store::StoredMessage &message, i32 utc_offset_minutes) {
    std::string line = std::format("[{:%H:%M}] ", local_seconds(message.time, utc_offset_minutes));
    line += display_name(message);
    line += message.from_self ? std::string("(你)") : "(" + std::to_string(message.sender_id) + ")";
    line += ": ";
    line += message.text;
    return line;
}

std::string fill_template(std::string_view text, const std::vector<std::pair<std::string_view, std::string_view>> &values) {
    std::string out;
    out.reserve(text.size());
    for (usize i = 0; i < text.size();) {
        bool replaced = false;
        if (text[i] == '{') {
            for (const auto &[key, value] : values) {
                if (text.substr(i + 1).starts_with(key) && text.substr(i + 1 + key.size()).starts_with('}')) {
                    out += value;
                    i += key.size() + 2;
                    replaced = true;
                    break;
                }
            }
        }
        if (!replaced) out.push_back(text[i++]);
    }
    return out;
}

PlannerRequest build_planner_request(const PlannerInput &input) {
    PlannerRequest request;
    provider::append_system(request.history, input.prompt);

    const std::unordered_set<i64> fresh(input.fresh.begin(), input.fresh.end());
    const auto kept = fit_lines(input.recent, input.log_bytes,
                                [&input](const store::StoredMessage &message) { return render_line(message, input.utc_offset_minutes); });

    std::string user = chat_facts(input.info, input.now, input.utc_offset_minutes);
    user += "\n\n聊天记录（编号在行首）：\n";
    bool marked = false;
    for (const auto index : kept) {
        const auto &message = input.recent[index];
        if (!marked && fresh.contains(message.message_id)) {
            user += "—— 以下是新消息 ——\n";
            marked = true;
        }
        request.ids.push_back(message.message_id);
        user += std::format("#{} {}\n", request.ids.size(), render_line(message, input.utc_offset_minutes));
    }
    user += input.addressed ? std::format("\n新消息里有人在叫{}，一般应该回应。", input.info.name) :
                              std::format("\n没有人直接叫{}，大多数时候她不需要说话。", input.info.name);
    provider::append_user(request.history, std::move(user));

    request.tools.push_back({
        .name = "reply",
        .description = "让她说话，回应其中一条消息",
        .input_schema = {.properties = {{"target", property("string", "要回应的消息编号，例如 \"#12\"")},
                                        {"quote", property("boolean", "是否引用那条消息")},
                                        {"style", property("string", "short | normal | long")},
                                        {"search", property("boolean", "是否需要联网搜索")},
                                        {"intent", property("string", "一句话描述她想表达的意思，不是台词")}},
                         .required = {"target", "style", "intent"}},
    });
    request.tools.push_back({
        .name = "stay_silent",
        .description = "这次她不说话",
        .input_schema = {.properties = {{"reason", property("string", "简短的理由")}}, .required = {"reason"}},
    });
    return request;
}

Plan parse_plan(const std::vector<provider::OutputItem> &items, const std::vector<i64> &ids) {
    for (const auto &item : items) {
        const auto *tool = std::get_if<provider::ToolCallItem>(&item);
        if (!tool) continue;
        const auto &input = tool->call.input;
        if (tool->call.name == "stay_silent") return {.reason = string_field(input, "reason")};
        if (tool->call.name != "reply") return {.reason = "unknown tool " + tool->call.name};

        const auto target = parse_target(string_field(input, "target"));
        if (!target || *target == 0 || *target > ids.size()) return {.reason = "invalid target " + string_field(input, "target")};
        const auto style = string_field(input, "style");
        return {
            .reply = true,
            .target = ids[*target - 1],
            .quote = bool_field(input, "quote"),
            .style = style == "long"   ? ReplyStyle::LONG :
                     style == "normal" ? ReplyStyle::NORMAL :
                                         ReplyStyle::SHORT,
            .search = bool_field(input, "search"),
            .intent = string_field(input, "intent"),
        };
    }
    return {.reason = "no tool call"};
}

provider::History build_writer_history(const WriterInput &input) {
    provider::History history;
    provider::append_system(history, input.persona);

    std::string developer = chat_facts(input.info, input.now, input.utc_offset_minutes);
    developer += "\n\n";
    developer += input.rules;
    developer += "\n";
    developer += input.style_override.empty() ? style_instruction(input.plan.style) : std::string_view(input.style_override);
    if (input.web_search) {
        developer +=
            "\n这次先用 web_search 查一下再说话。查完只挑最关键的一点，用一句大白话说出来，不报数字、日期和来源，不贴链接，除非别人要。";
    }
    provider::append_developer(history, std::move(developer));

    const auto turn_text = [](const store::StoredMessage &message) {
        return message.from_self ? message.text : display_name(message) + ": " + message.text;
    };
    const auto kept = fit_lines(input.recent, input.log_bytes, turn_text);

    for (const auto index : kept) {
        const auto &message = input.recent[index];
        const auto role = message.from_self ? provider::Role::ASSISTANT : provider::Role::USER;
        if (history.size() > 2 && history.back().role == role) {
            std::get<provider::TextPart>(history.back().parts.back()).text += "\n" + turn_text(message);
            continue;
        }
        if (role == provider::Role::ASSISTANT && history.size() == 2) provider::append_user(history, "（更早的聊天记录略）");
        history.push_back({.role = role, .parts = {provider::TextPart{.text = turn_text(message)}}});
    }

    std::string note = std::format("【你要回应{}的这条：「{}」", display_name(input.target), input.target.text);
    if (!input.plan.intent.empty()) note += std::format("。你想表达的意思：{}", input.plan.intent);
    note += "】\n现在，你说：";
    if (history.size() > 2 && history.back().role == provider::Role::USER) {
        std::get<provider::TextPart>(history.back().parts.back()).text += "\n\n" + note;
    } else {
        provider::append_user(history, std::move(note));
    }
    return history;
}

std::string strip_markdown(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool line_start = true;
    for (usize i = 0; i < text.size();) {
        if (line_start) {
            // "## heading" -> "heading"
            usize hashes = 0;
            while (i + hashes < text.size() && text[i + hashes] == '#') ++hashes;
            if (hashes > 0 && hashes <= 6 && i + hashes < text.size() && text[i + hashes] == ' ') i += hashes + 1;
        }
        if (text.substr(i).starts_with("**") || text.substr(i).starts_with("__")) {
            i += 2;
            line_start = false;
            continue;
        }
        line_start = text[i] == '\n';
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

std::string strip_asides(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (usize i = 0; i < text.size();) {
        const bool full = text.substr(i).starts_with("（");
        if (full || text[i] == '(') {
            const auto open = full ? std::string_view("（").size() : 1;
            const auto close = text.find(full ? "）" : ")", i + open);
            const auto inner = close == std::string_view::npos ? std::string_view{} : text.substr(i + open, close - i - open);
            // Only asides that contain CJK text, like MaiBot; "(C++)" stays.
            const bool cjk = std::ranges::any_of(inner, [](char c) { return static_cast<unsigned char>(c) >= 0xE3; });
            if (close != std::string_view::npos && cjk) {
                i = close + (full ? std::string_view("）").size() : 1);
                continue;
            }
        }
        if (text.substr(i).starts_with("——")) {
            out += "，";
            i += std::string_view("——").size();
            continue;
        }
        out.push_back(text[i++]);
    }
    return out;
}

std::string clean_reply(std::string_view text, const ChatInfo &info, const std::vector<std::string> &aliases, usize max_bytes) {
    auto cleaned = strip_asides(strip_markdown(strip_speaker_label(text, info, aliases)));
    if (cleaned.size() <= max_bytes) return cleaned;
    constexpr std::string_view k_ellipsis = "……";
    auto bounded = bounded_utf8(cleaned, max_bytes > k_ellipsis.size() ? max_bytes - k_ellipsis.size() : 0);
    bounded += k_ellipsis;
    return bounded;
}

} // namespace lilia::chat
