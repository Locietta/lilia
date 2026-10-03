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
    for (const auto &alias : aliases) {
        if (alias.empty() || alias.size() > text.size()) continue;
        for (usize start = 0; start + alias.size() <= text.size(); ++start) {
            bool matched = true;
            for (usize i = 0; i < alias.size(); ++i) {
                if (fold(text[start + i]) != fold(alias[i])) {
                    matched = false;
                    break;
                }
            }
            if (matched) return true;
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
