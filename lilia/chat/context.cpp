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

glz::generic string_schema() {
    glz::generic schema;
    std::ignore = glz::read_json(schema, R"({"type": "string"})");
    return schema;
}

provider::SchemaProperty property(std::string type, std::string description) {
    return {.type = std::move(type), .description = std::move(description)};
}

provider::SchemaProperty string_array_property(std::string description) {
    provider::SchemaProperty result{.type = "array", .description = std::move(description)};
    result.items.emplace(string_schema());
    return result;
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
        case ReplyStyle::SHORT: return "这次就一句短话，一两个词也行。";
        case ReplyStyle::NORMAL: return "这次一两句话。";
        case ReplyStyle::LONG: return "这次是有意发一整条长消息，可以分几行把事情讲清楚，还是聊天的口气，三百字以内。";
    }
    return "";
}

/// "@阿紫 芒果" -> "芒果": a typed mention is plain text in QQ, not a real @.
std::string_view strip_leading_mention(std::string_view text) {
    while (text.starts_with('@')) {
        const auto space = text.find_first_of(" \n");
        if (space == std::string_view::npos || space > 40) break;
        text = trim(text.substr(space + 1));
    }
    return text;
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
    if (input.just_replied) {
        const auto it = std::ranges::find(request.ids, *input.just_replied);
        user += it == request.ids.end() ? "\n她刚回复完一条消息。" :
                                          std::format("\n她刚回复了 #{}。", static_cast<usize>(it - request.ids.begin()) + 1);
        user += "新消息里其余的她还没回应过：看看有没有人在跟她说话、或者还有她想接的，没有就不说话，别为了说而说。";
    } else {
        user += input.addressed ? std::format("\n新消息里有人在叫{}，一般应该回应。", input.info.name) :
                                  std::format("\n没有人直接叫{}。她可以只是看着，也可以在有话想说的时候插一句。", input.info.name);
    }
    if (!input.people.empty()) {
        // MaiBot's person profiles: background for the planner only.
        user += "\n\n群友资料（她已经知道这些，别让她再问一遍；内部参考，别逐字复述；和眼前的聊天冲突时以聊天为准）：\n" + input.people;
    }
    if (!input.recall.empty()) {
        user += "\n\n聊天回想（以前聊过的相关内容，内部参考；自然相关时才用，别生硬复述）：\n" + input.recall;
    }
    if (!input.decisions.empty()) {
        user += std::format("\n\n{}最近几次的判断（旧到新）：\n", input.info.name);
        for (const auto &decision : input.decisions) user += "- " + decision + "\n";
        user += "别重复她刚说过的意思。";
    }
    provider::append_user(request.history, std::move(user));

    request.tools.push_back({
        .name = "reply",
        .description = "让她说话，回应其中一条消息",
        .input_schema = {.properties = {{"target", property("string", "要回应的消息编号，例如 \"#12\"")},
                                        {"quote", property("boolean", "是否引用那条消息")},
                                        {"style", property("string", "short | normal | long")},
                                        {"search", property("boolean", "是否需要联网搜索")},
                                        {input.reference_handoff ? "brief" : "intent",
                                         input.reference_handoff ? property("string", "给她的参考信息，不是台词，也不规定她怎么说") :
                                                                   property("string", "一句话描述她想表达的意思，不是台词")}},
                         .required = {"target", "style", input.reference_handoff ? "brief" : "intent"}},
    });
    if (input.memory_search) {
        request.tools.push_back({
            .name = "search_history",
            .description = "搜索这个群以前的聊天记录，结果会告诉你，然后你再决定 reply 还是 stay_silent",
            .input_schema = {.properties = {{"query",
                                             property("string", "要找的内容，写成一句完整的问题或描述，例如\"谁说过对芒果过敏\"")}},
                             .required = {"query"}},
        });
    }
    request.tools.push_back({
        .name = "stay_silent",
        .description = "这次她不说话",
        .input_schema = {.properties = {{"reason", property("string", "简短的理由")}}, .required = {"reason"}},
    });
    return request;
}

std::string_view memory_rule(bool enabled) {
    if (!enabled) return "";
    return "- 有人问到以前的事（\"上次\"\"之前\"\"来着\"\"还记得吗\"、过去的约定和时间、某人以前说过的话），而上面的聊天记录里没有时，"
           "一定先用 search_history 查，不要直接让她说不记得。寒暄、接梗、看最近的聊天就能回的，不要查。\n";
}

std::string_view handoff_rule(bool reference) {
    if (reference) {
        return "- brief：给她的参考信息，像旁观者的笔记：大家在聊什么、她对这件事大概什么态度、回复可能用得上的事实。"
               "不要写台词，也不要规定她具体说什么、怎么说，那由她自己决定。";
    }
    return "- intent：用一句短话（十来个字）写她的态度或想说的点，例如\"吐槽这集作画崩了\"或\"问他抽到没有\"。"
           "不要写成台词，也不要写成一套论证或建议（\"认同……但提醒……建议……\"）。";
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
            .brief = string_field(input, "brief"),
        };
    }
    return {.reason = "no tool call"};
}

std::string quoted_phrase(std::string_view style) {
    const auto open = style.find('"');
    if (open == std::string_view::npos) return {};
    const auto close = style.find('"', open + 1);
    if (close == std::string_view::npos || close == open + 1) return {};
    return std::string(style.substr(open + 1, close - open - 1));
}

provider::History build_writer_history(const WriterInput &input) {
    provider::History history;
    provider::append_system(history, input.persona);

    std::string developer = chat_facts(input.info, input.now, input.utc_offset_minutes);
    developer += "\n\n";
    developer += input.rules;
    developer += "\n";
    developer += input.style_override.empty() ? style_instruction(input.plan.style) : std::string_view(input.style_override);
    if (!input.voice.empty()) {
        developer += "\n\n群里的人平时会说这样的话，只是让你找找大家说话的口气，不用照搬：";
        for (usize i = 0; i < input.voice.size(); ++i) developer += (i ? "、" : "") + std::format("「{}」", input.voice[i]);
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
    if (!input.reference.empty()) {
        // The planner's intent predates the search; after it she knows.
        note += std::format("。你刚去查过了，情况如下：{}\n"
                            "你已经知道是怎么回事了，不要再问真假或者出处。挑最关键的一点或者你的看法，用一句大白话说，"
                            "不报数字、日期和来源，不贴链接，别照抄】",
                            input.reference);
    } else if (!input.plan.brief.empty()) {
        note += std::format("】\n【回复信息参考：{}\n可以参考，但视情况而定，不用完全遵守】", input.plan.brief);
    } else {
        if (!input.plan.intent.empty()) note += std::format("。你想表达的意思：{}", input.plan.intent);
        note += "】";
    }
    if (!input.known.empty()) {
        note += std::format("\n【你知道的关于{}的事，用得上就自然地用，别逐条复述，也别再问一遍：{}】", display_name(input.target),
                            input.known);
    }
    if (!input.remembered.empty()) {
        note += std::format("\n【你想起来的以前的聊天，用得上就自然地用，别照抄，也别说你查过记录】\n{}", input.remembered);
    }
    if (!input.previous_reply.empty()) {
        note += std::format("\n【你刚刚已经回复过这条消息，当时说的是：「{}」。这次是补充，别重复之前的话】", input.previous_reply);
    }
    if (!input.habits.empty()) {
        // MaiBot's wording for picked expressions.
        note += "\n【表达习惯参考，请视情况自然地使用】";
        for (const auto &habit : input.habits) note += "\n- " + habit;
    }
    note += "\n现在，你说：";
    if (history.size() > 2 && history.back().role == provider::Role::USER) {
        std::get<provider::TextPart>(history.back().parts.back()).text += "\n\n" + note;
    } else {
        provider::append_user(history, std::move(note));
    }
    return history;
}

provider::History build_search_history(const ChatInfo &info, const store::StoredMessage &target, const Plan &plan, i64 now,
                                       i32 utc_offset_minutes) {
    provider::History history;
    provider::append_system(history, std::format("你是搜索助手。现在是 {}。用 web_search 搜索最新信息，然后用三四句中文客观总结查到的要点"
                                                 "（发生了什么、什么时候、关键细节和争议），不评论，不用 Markdown。",
                                                 format_local_time(now, utc_offset_minutes)));
    std::string user = std::format("群聊里{}说：「{}」", display_name(target), target.text);
    if (!plan.intent.empty()) user += std::format("\n{}想知道的：{}", info.name, plan.intent);
    if (!plan.brief.empty()) user += std::format("\n背景：{}", plan.brief);
    user += "\n请先搜索再总结。";
    provider::append_user(history, std::move(user));
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
    auto cleaned = strip_asides(strip_markdown(strip_leading_mention(strip_speaker_label(text, info, aliases))));
    if (cleaned.size() <= max_bytes) return cleaned;
    constexpr std::string_view k_ellipsis = "……";
    auto bounded = bounded_utf8(cleaned, max_bytes > k_ellipsis.size() ? max_bytes - k_ellipsis.size() : 0);
    bounded += k_ellipsis;
    return bounded;
}

std::string strip_transcript(std::string_view text, const std::vector<store::StoredMessage> &recent) {
    const auto after_colon = [](std::string_view rest) -> std::optional<std::string_view> {
        if (rest.starts_with(':')) return trim(rest.substr(1));
        if (rest.starts_with("：")) return trim(rest.substr(std::string_view("：").size()));
        return std::nullopt;
    };
    std::string out;
    while (!text.empty()) {
        const auto newline = text.find('\n');
        auto line = trim(text.substr(0, newline));
        text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);
        bool labelled = false;
        // "10001:" — a QQ number used as a label.
        const auto digits = line.find_first_not_of("0123456789");
        if (digits >= 5 && digits != std::string_view::npos) {
            if (const auto rest = after_colon(line.substr(digits))) {
                line = *rest;
                labelled = true;
            }
        }
        for (const auto &message : recent) {
            if (labelled || message.from_self) continue;
            const auto name = display_name(message);
            if (name.empty() || !line.starts_with(name)) continue;
            if (const auto rest = after_colon(line.substr(name.size()))) {
                line = *rest;
                labelled = true;
            }
        }
        if (line.empty()) continue;
        if (labelled && std::ranges::any_of(recent, [&](const store::StoredMessage &m) { return trim(m.text) == line; })) continue;
        if (!out.empty()) out += '\n';
        out += line;
    }
    return out;
}

} // namespace lilia::chat
