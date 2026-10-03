#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <lilia/provider/history.h>
#include <lilia/store/store.h>

namespace lilia::chat {

struct ChatInfo {
    store::ChatKey chat;
    std::string chat_name; ///< group name, or the peer's nickname in private chats
    i64 self_id = 0;
    std::string self_name;
};

struct ContextRequest {
    std::string persona;
    ChatInfo info;
    /// Recent chat log, oldest first. May include the triggers themselves.
    std::vector<store::StoredMessage> recent;
    /// Messages lilia should answer, oldest first.
    std::vector<store::StoredMessage> triggers;
    i64 now = 0; ///< unix seconds
    i32 utc_offset_minutes = 480;
    usize log_bytes = 12'000;
    /// The model can search the web (adds usage guidance to the rules).
    bool web_search = false;
};

/// "[HH:MM] name(qq): text", with "(你)" marking lilia's own lines.
std::string render_line(const store::StoredMessage &message, i32 utc_offset_minutes);

/// "YYYY-MM-DD HH:MM" in the given offset.
std::string format_local_time(i64 unix_seconds, i32 utc_offset_minutes);

/// Unix seconds of the most recent local midnight.
i64 local_day_start(i64 unix_seconds, i32 utc_offset_minutes);

/// SYSTEM persona, DEVELOPER runtime facts and reply rules, then one USER
/// turn holding the recent log (newest lines kept within the byte budget)
/// and the messages to answer.
provider::History build_context(const ContextRequest &request);

/// Removes Markdown that QQ shows literally: heading markers and **bold**.
std::string strip_markdown(std::string_view text);

/// Cleans a model answer for QQ: trims, drops a leading speaker label the
/// model may copy from the log format, strips Markdown and bounds the length.
std::string clean_reply(std::string_view text, const ChatInfo &info, const std::vector<std::string> &aliases, usize max_bytes);

} // namespace lilia::chat
