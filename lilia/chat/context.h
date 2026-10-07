#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <lilia/provider/common.h>
#include <lilia/provider/history.h>
#include <lilia/store/store.h>

namespace lilia::chat {

struct ChatInfo {
    store::ChatKey chat;
    std::string chat_name; ///< group name, or the peer's nickname in private chats
    std::string name;      ///< lilia's persona name, e.g. 莉莉娅
    i64 self_id = 0;
    std::string self_name; ///< the account's QQ nickname
};

enum struct ReplyStyle {
    SHORT,
    NORMAL,
    LONG,
};

/// The planner's decision for one turn.
struct Plan {
    bool reply = false;
    i64 target = 0; ///< message id lilia answers
    bool quote = false;
    ReplyStyle style = ReplyStyle::SHORT;
    bool search = false;
    std::string intent;
    std::string reason; ///< why she stays silent
};

struct PlannerInput {
    std::string prompt; ///< planner.md with {name} and {persona} filled in
    ChatInfo info;
    /// Recent chat log, oldest first; ends with the new messages.
    std::vector<store::StoredMessage> recent;
    /// Ids of messages the planner has not seen before.
    std::vector<i64> fresh;
    bool addressed = false;
    i64 now = 0;
    i32 utc_offset_minutes = 480;
    usize log_bytes = 12'000;
};

struct PlannerRequest {
    provider::History history;
    std::vector<provider::ToolDefinition> tools;
    /// ids[n - 1] is the message id shown as "#n".
    std::vector<i64> ids;
};

PlannerRequest build_planner_request(const PlannerInput &input);

/// Reads the planner's tool call. No call, an unknown tool or an invalid
/// target means silence.
Plan parse_plan(const std::vector<provider::OutputItem> &items, const std::vector<i64> &ids);

struct WriterInput {
    std::string persona;
    std::string rules; ///< writer.md
    ChatInfo info;
    std::vector<store::StoredMessage> recent;
    store::StoredMessage target;
    Plan plan;
    /// Occasional one-off style that replaces the planner's style.
    std::string style_override;
    i64 now = 0;
    i32 utc_offset_minutes = 480;
    usize log_bytes = 12'000;
    bool web_search = false;
};

/// SYSTEM persona; DEVELOPER facts, rules and style; then the chat log as
/// alternating turns (lilia's own lines are assistant turns) ending with a
/// note that it is her turn to answer `target`.
provider::History build_writer_history(const WriterInput &input);

/// "[HH:MM] name(qq): text", with "(你)" marking lilia's own lines.
std::string render_line(const store::StoredMessage &message, i32 utc_offset_minutes);

/// "YYYY-MM-DD HH:MM" in the given offset.
std::string format_local_time(i64 unix_seconds, i32 utc_offset_minutes);

/// Unix seconds of the most recent local midnight.
i64 local_day_start(i64 unix_seconds, i32 utc_offset_minutes);

/// Replaces every "{key}" with its value.
std::string fill_template(std::string_view text, const std::vector<std::pair<std::string_view, std::string_view>> &values);

/// Removes Markdown that QQ shows literally: heading markers and **bold**.
std::string strip_markdown(std::string_view text);

/// Removes bracketed asides containing CJK text ("（笑" stays without a
/// closing bracket) and turns "——" into a comma.
std::string strip_asides(std::string_view text);

/// Cleans a model answer for QQ: trims, drops a leading speaker label the
/// model may copy from the log format, strips Markdown and bounds the length.
std::string clean_reply(std::string_view text, const ChatInfo &info, const std::vector<std::string> &aliases, usize max_bytes);

} // namespace lilia::chat
