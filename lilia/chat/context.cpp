#include "context.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <variant>

#include <lilia/chat/delivery.h>
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

} // namespace

std::string display_name(const store::StoredMessage &message) {
    return message.sender_name.empty() ? std::to_string(message.sender_id) : message.sender_name;
}

namespace {

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

std::string_view style_instruction(ReplyStyle style) {
    switch (style) {
        // MaiBot's 简短表达: "允许句子残缺，奇怪表达，倒装，省略……"
        case ReplyStyle::SHORT: return "这次就一句短话，一两个词也行。允许句子残缺、倒装、省略，怎么省力怎么说。";
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

std::string_view without_quote(std::string_view text) {
    text = trim(text);
    if (text.starts_with("[回复")) {
        if (const auto close = text.find("] "); close != std::string_view::npos) return trim(text.substr(close + 2));
        if (text == "[回复]") return {};
    }
    return text;
}

namespace {

/// The first `count` code points of a message, without what it quotes, on
/// one line ("…" when cut).
std::string excerpt(std::string_view text, usize count) {
    text = without_quote(text);
    std::string out;
    usize kept = 0;
    for (usize i = 0; i < text.size() && kept < count; ++kept) {
        usize next = i + 1;
        while (next < text.size() && (static_cast<unsigned char>(text[next]) & 0xC0) == 0x80) ++next;
        out.append(text.substr(i, next - i));
        i = next;
    }
    if (out.size() < text.size()) out += "…";
    std::ranges::replace(out, '\n', ' ');
    return out;
}

/// The planner's "#18345" (a row number the writer never sees) as the line
/// itself: 「阿紫: 第三集…」.
std::string resolve_rows(std::string_view text, const std::vector<store::StoredMessage> &recent) {
    constexpr usize k_excerpt_code_points = 12;
    std::string out;
    for (usize i = 0; i < text.size();) {
        usize end = i + 1;
        while (text[i] == '#' && end < text.size() && text[end] >= '0' && text[end] <= '9') ++end;
        if (end == i + 1) {
            out.push_back(text[i++]);
            continue;
        }
        i64 row = 0;
        for (usize k = i + 1; k < end; ++k) row = row * 10 + (text[k] - '0');
        if (const auto it = std::ranges::find(recent, row, &store::StoredMessage::row_id); it != recent.end()) {
            out +=
                std::format("「{}: {}」", it->from_self ? std::string("你") : display_name(*it), excerpt(it->text, k_excerpt_code_points));
        } else {
            out += "那条消息";
        }
        i = end < text.size() && text[end] == ' ' ? end + 1 : end; // "#12 在问" was spaced for the number
    }
    return out;
}

} // namespace

std::string quote_prefix(const store::StoredMessage *quoted, std::string_view self_name) {
    constexpr usize k_excerpt_code_points = 20;
    if (!quoted) return "[回复] ";
    // Only the quoted line itself, not what it quoted in turn.
    const auto name = quoted->from_self ? std::string(self_name) : display_name(*quoted);
    return std::format("[回复 {}: {}] ", name, excerpt(quoted->text, k_excerpt_code_points));
}

std::string render_line(const store::StoredMessage &message, i32 utc_offset_minutes) {
    std::string line = std::format("[{:%H:%M}] ", local_seconds(message.time, utc_offset_minutes));
    if (is_notice(message)) return line + "（群通知）" + message.text;
    line += display_name(message);
    line += message.from_self ? std::string("(她自己)") : "(" + std::to_string(message.sender_id) + ")";
    line += ": ";
    line += message.text;
    return line;
}

std::string chat_place(const ChatInfo &info) {
    std::string facts =
        info.chat.type == "group" ? std::format("这是 QQ 群「{}」。", info.chat_name) : std::format("这是和「{}」的私聊。", info.chat_name);
    facts += std::format("{}在这里的名字是「{}」，别人 @ 她时会显示成「@{}」。", info.name, info.self_name, info.self_name);
    return facts;
}

std::string clock_line(i64 now, i32 utc_offset_minutes) {
    return std::format("现在是 {}（{}）。", format_local_time(now, utc_offset_minutes), offset_label(utc_offset_minutes));
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

std::string_view sticker_rule(bool enabled) {
    if (!enabled) return "";
    return "- 群里大家常用表情包回应。她想表达的主要是情绪或反应（笑死、无语、震惊、赞同、收到、委屈、看戏、晚安）时，发一张表情包"
           "往往比打字自然，也比打\"确实\"\"草\"这种一两个字的反应好。先用 find_sticker 按想表达的感觉找几张，挑一张贴合语气的用 "
           "send_sticker 发；都不合适就别发，或者打字。说完话想补个情绪，也可以接着发一张。别人发的表情包好玩，也可以用 repeat "
           "复读那一张。认真回答问题、说正事时不用表情包；刚发过一张就隔一会儿，也别总发同一张。\n";
}

std::string memory_rule(bool history, bool people) {
    std::string rule;
    if (history) {
        rule += "- 有人问到以前的事（\"上次\"\"之前\"\"来着\"\"还记得吗\"、过去的约定和时间、某人以前说过的话），而上面的聊天记录里没有时，"
                "一定先用 search_history 查，不要直接让她说不记得。寒暄、接梗、看最近的聊天就能回的，不要查。\n";
    }
    if (people) {
        rule += "- 要接某个群友自己的事（近况、工作、爱好、住哪）时，可以先用 look_up_person 看看她以前记下的关于这个人的事，"
                "免得问人家早就说过的。\n";
    }
    return rule;
}

std::string_view handoff_rule(bool reference) {
    if (reference) {
        return "- brief：给她的参考信息，像旁观者的笔记：大家在聊什么、她对这件事大概什么态度、回复可能用得上的事实。"
               "不要写台词，也不要规定她具体说什么、怎么说，那由她自己决定。";
    }
    return "- intent：用一句短话（十来个字）写她的态度或想说的点，例如\"吐槽这集作画崩了\"或\"问他抽到没有\"。"
           "不要写成台词，也不要写成一套论证或建议（\"认同……但提醒……建议……\"）。";
}

provider::History build_writer_history(const WriterInput &input) {
    // Persona and rules never change, then the anchored log: both cacheable.
    // Where, when, the style and every note come in the final user part.
    provider::History history;
    provider::append_system(history, input.persona);
    provider::append_system(history, input.rules);

    const auto turn_text = [](const store::StoredMessage &message) {
        if (is_notice(message)) return "（群通知）" + message.text;
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
    if (history.size() > 2) std::get<provider::TextPart>(history.back().parts.back()).cache = true;

    std::string note = chat_place(input.info) + clock_line(input.now, input.utc_offset_minutes) + "\n";
    note += style_instruction(input.plan.style);
    note += std::format("\n【你要回应{}的这条：「{}」", display_name(input.target), input.target.text);
    if (input.plan.to_her == "no") note += "（这条不是对你说的，你是插一句：别装作对方在问你）";
    if (!input.plan.brief.empty()) {
        note += std::format("】\n【回复信息参考：{}\n可以参考，但视情况而定，不用完全遵守】", resolve_rows(input.plan.brief, input.recent));
    } else {
        if (!input.plan.intent.empty()) note += std::format("。你想表达的意思：{}", resolve_rows(input.plan.intent, input.recent));
        note += "】";
    }
    if (!input.reference.empty()) {
        // What the planner looked up online before deciding to reply.
        note += std::format("\n【你刚去查过了，情况如下：{}\n"
                            "你已经知道是怎么回事了，不要再问真假或者出处。挑最关键的一点或者你的看法，用一句大白话说，"
                            "不报数字、日期和来源，不贴链接，别照抄】",
                            input.reference);
    }
    for (const auto &person : input.known) {
        note += std::format("\n【你知道的关于{}的事，用得上就自然地用，别逐条复述，也别再问一遍：{}】", person.name, person.facts);
    }
    if (!input.presence.empty()) note += std::format("\n【{}】", input.presence);
    if (!input.self_notes.empty()) {
        note += std::format("\n【你以前说过的关于你自己的事，说话要和这些一致、别自相矛盾（同一件事以日期新的为准），不用特意提起】\n{}",
                            input.self_notes);
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
        // Its own part: the log before it stays a cacheable prefix.
        history.back().parts.push_back(provider::TextPart{.text = std::move(note)});
    } else {
        provider::append_user(history, std::move(note));
    }
    return history;
}

provider::History build_search_history(std::string_view query, i64 now, i32 utc_offset_minutes) {
    provider::History history;
    provider::append_system(history, std::format("你是搜索助手。现在是 {}。用 web_search 搜索最新信息，然后用三四句中文客观总结查到的要点"
                                                 "（发生了什么、什么时候、关键细节和争议），不评论，不用 Markdown。查不到就直说查不到。",
                                                 format_local_time(now, utc_offset_minutes)));
    provider::append_user(history, std::format("要查的：{}\n请先搜索再总结。", query));
    return history;
}

std::string_view search_rule(bool enabled) {
    if (!enabled) return "";
    return "- 有人提到具体的新闻、发布、事件、数据或者说法，她不确定真假、不了解细节，或者被问到要查才知道的事（天气、价格、时间、"
           "最新消息）时，先用 web_search 查一下，看到结果再决定怎么回。纯闲聊、接梗不要查。\n";
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

namespace {

/// `text` cut to `max_bytes` with an ellipsis, when longer.
std::string bound_bytes(std::string text, usize max_bytes) {
    if (text.size() <= max_bytes) return text;
    constexpr std::string_view k_ellipsis = "……";
    auto bounded = bounded_utf8(text, max_bytes > k_ellipsis.size() ? max_bytes - k_ellipsis.size() : 0);
    bounded += k_ellipsis;
    return bounded;
}

} // namespace

std::string clean_reply(std::string_view text, const ChatInfo &info, const std::vector<std::string> &aliases, usize max_bytes) {
    return bound_bytes(strip_asides(strip_markdown(strip_leading_mention(strip_speaker_label(text, info, aliases)))), max_bytes);
}

Repaired repair_reply(std::string_view raw, const std::vector<store::StoredMessage> &recent, const ChatInfo &info,
                      const std::vector<std::string> &aliases, const ReplyLimits &limits) {
    Repaired out{.text = std::string(trim(raw))};
    const auto apply = [&](std::string_view kind, std::string next) {
        if (trim(next) != trim(out.text)) out.repairs.push_back(kind);
        out.text = std::move(next);
    };
    apply("transcript", strip_transcript(out.text, recent));
    apply("label", std::string(strip_speaker_label(out.text, info, aliases)));
    apply("mention", std::string(strip_leading_mention(out.text)));
    apply("markdown", strip_markdown(out.text));
    apply("aside", strip_asides(out.text));
    apply("bytes", bound_bytes(out.text, limits.max_bytes));
    apply("length", fit_length(out.text, limits.max_code_points));
    std::vector<char32_t> removed;
    apply("emoji", strip_emoji(out.text, limits.kept_emoji, removed));
    return out;
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
