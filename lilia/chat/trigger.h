#pragma once

#include <string_view>
#include <vector>

#include <lilia/config.h>
#include <lilia/onebot/types.h>

namespace lilia::chat {

enum struct TriggerReason {
    NONE,
    MENTION, ///< @-mentioned the bot
    REPLY,   ///< quoted one of the bot's messages
    ALIAS,   ///< called the bot by name
    PRIVATE, ///< any message in an allowed private chat
};

std::string_view to_string(TriggerReason reason) noexcept;

/// Whether lilia may log and answer this chat at all.
bool chat_allowed(const onebot::MessageEvent &event, const ChatConfig &config);

/// ASCII letters compare case-insensitively; other bytes must match exactly.
bool contains_alias(std::string_view text, const std::vector<std::string> &aliases);

/// `replies_to_self` is whether the quoted message (if any) was sent by lilia.
TriggerReason classify(const onebot::MessageEvent &event, i64 self_id, bool replies_to_self, const ChatConfig &config);

/// True for "/lilia" admin commands, returning the argument text after it.
bool parse_admin_command(std::string_view text, std::string_view &arguments);

} // namespace lilia::chat
