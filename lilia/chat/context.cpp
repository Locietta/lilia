#include "context.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <unordered_set>

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

/// Removes "[HH:MM] " and "name(qq): " prefixes the model may imitate.
std::string_view strip_speaker_label(std::string_view text, const ChatInfo &info, const std::vector<std::string> &aliases) {
    text = trim(text);
    if (text.starts_with('[')) {
        const auto close = text.find(']');
        if (close != std::string_view::npos && close <= 20) text = trim(text.substr(close + 1));
    }
    std::vector<std::string_view> names{info.self_name};
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
    line += message.sender_name.empty() ? std::to_string(message.sender_id) : message.sender_name;
    line += message.from_self ? std::string("(你)") : "(" + std::to_string(message.sender_id) + ")";
    line += ": ";
    line += message.text;
    return line;
}

provider::History build_context(const ContextRequest &request) {
    provider::History history;
    provider::append_system(history, request.persona);

    const bool group = request.info.chat.type == "group";
    std::string facts = std::format("当前时间：{}（{}）\n", format_local_time(request.now, request.utc_offset_minutes),
                                    offset_label(request.utc_offset_minutes));
    if (group) {
        facts += std::format("你在 QQ 群「{}」（群号 {}）里聊天。", request.info.chat_name, request.info.chat.id);
    } else {
        facts += std::format("你在和「{}」（QQ {}）私聊。", request.info.chat_name, request.info.chat.id);
    }
    facts += std::format("你的 QQ 号是 {}，昵称是「{}」。\n\n", request.info.self_id, request.info.self_name);
    facts += "聊天记录每行的格式是「[时间] 昵称(QQ号): 内容」，标着「(你)」的是你自己说过的话。"
             "图片、表情等非文字内容以方括号占位，例如「[图片]」，你看不到它们的具体内容。\n\n"
             "回复规则：\n"
             "- 只回应最后列出的「需要你回复的消息」，聊天记录只作背景参考。\n"
             "- 直接给出要发送的内容，不要加时间、昵称或「你:」之类的前缀。\n"
             "- 这是即时聊天，像群友一样自然、简短地说话；不要使用 Markdown 标题、列表或代码块。\n"
             "- 不知道的事情就说不知道，不要编造。";
    if (request.web_search) {
        facts += "\n- 你可以用 web_search 联网搜索。只在需要最新信息或核实事实时搜索，闲聊不要搜。"
                 "搜索后用几句话说结论，需要时附上一个最相关的来源链接，不要逐条罗列搜索结果。";
    }
    provider::append_developer(history, std::move(facts));

    std::unordered_set<i64> trigger_ids;
    for (const auto &trigger : request.triggers) trigger_ids.insert(trigger.message_id);

    // Walk newest to oldest so the budget drops the oldest lines first.
    std::vector<std::string> lines;
    usize used = 0;
    for (auto it = request.recent.rbegin(); it != request.recent.rend(); ++it) {
        if (trigger_ids.contains(it->message_id) && !it->from_self) continue;
        auto line = render_line(*it, request.utc_offset_minutes);
        if (used + line.size() + 1 > request.log_bytes) break;
        used += line.size() + 1;
        lines.push_back(std::move(line));
    }
    std::ranges::reverse(lines);

    std::string user;
    if (!lines.empty()) {
        user += "最近的聊天记录：\n";
        for (const auto &line : lines) {
            user += line;
            user += '\n';
        }
        user += '\n';
    }
    user += request.triggers.size() > 1 ? "需要你回复的消息（按时间顺序）：\n" : "需要你回复的消息：\n";
    for (const auto &trigger : request.triggers) {
        user += render_line(trigger, request.utc_offset_minutes);
        user += '\n';
    }
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

std::string clean_reply(std::string_view text, const ChatInfo &info, const std::vector<std::string> &aliases, usize max_bytes) {
    auto cleaned = strip_markdown(strip_speaker_label(text, info, aliases));
    if (cleaned.size() <= max_bytes) return cleaned;
    constexpr std::string_view k_ellipsis = "……";
    auto bounded = bounded_utf8(cleaned, max_bytes > k_ellipsis.size() ? max_bytes - k_ellipsis.size() : 0);
    bounded += k_ellipsis;
    return bounded;
}

} // namespace lilia::chat
