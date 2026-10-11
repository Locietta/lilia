#include "trigger.h"

#include <algorithm>

#include <lilia/onebot/message.h>

namespace lilia::chat {

namespace {

char fold(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\n' || text.front() == '\r' || text.front() == '\t'))
        text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n' || text.back() == '\r' || text.back() == '\t'))
        text.remove_suffix(1);
    return text;
}

} // namespace

std::string_view to_string(TriggerReason reason) noexcept {
    switch (reason) {
        case TriggerReason::NONE: return "none";
        case TriggerReason::MENTION: return "mention";
        case TriggerReason::REPLY: return "reply";
        case TriggerReason::ALIAS: return "alias";
        case TriggerReason::PRIVATE: return "private";
    }
    return "unknown";
}

bool chat_allowed(const onebot::MessageEvent &event, const ChatConfig &config) {
    if (event.is_group()) return std::ranges::contains(config.groups, *event.group_id);
    if (event.message_type == "private") return std::ranges::contains(config.private_users, event.user_id);
    return false;
}

bool contains_alias(std::string_view text, const std::vector<std::string> &aliases) {
    const auto alnum = [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    const auto space = [](char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; };
    for (const auto &alias : aliases) {
        if (alias.empty() || alias.size() > text.size()) continue;
        const bool ascii = std::ranges::all_of(alias, [](char c) { return static_cast<unsigned char>(c) < 0x80; });
        for (usize start = 0; start + alias.size() <= text.size(); ++start) {
            bool matched = true;
            for (usize i = 0; i < alias.size(); ++i) {
                if (fold(text[start + i]) != fold(alias[i])) {
                    matched = false;
                    break;
                }
            }
            if (!matched) continue;
            if (ascii) {
                // A whole word ("lilia在吗"), not part of one ("Ophilia") ...
                const usize end = start + alias.size();
                if ((start > 0 && alnum(text[start - 1])) || (end < text.size() && alnum(text[end]))) continue;
                // ... nor of a link or path ("github.com/x/lilia").
                usize first = start;
                while (first > 0 && !space(text[first - 1])) --first;
                usize last = end;
                while (last < text.size() && !space(text[last])) ++last;
                if (text.substr(first, last - first).contains('/')) continue;
            }
            return true;
        }
    }
    return false;
}

TriggerReason classify(const onebot::MessageEvent &event, i64 self_id, bool replies_to_self, const ChatConfig &config) {
    if (event.user_id == self_id || !chat_allowed(event, config)) return TriggerReason::NONE;
    if (!event.is_group()) return TriggerReason::PRIVATE;
    if (onebot::mentions(event.message, self_id)) return TriggerReason::MENTION;
    if (replies_to_self) return TriggerReason::REPLY;
    if (contains_alias(onebot::text_only(event.message), config.aliases)) return TriggerReason::ALIAS;
    return TriggerReason::NONE;
}

bool parse_admin_command(std::string_view text, std::string_view &arguments) {
    text = trim(text);
    constexpr std::string_view k_prefix = "/lilia";
    if (!text.starts_with(k_prefix)) return false;
    const auto rest = text.substr(k_prefix.size());
    if (!rest.empty() && rest.front() != ' ') return false;
    arguments = trim(rest);
    return true;
}

} // namespace lilia::chat
