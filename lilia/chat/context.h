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
    /// "reference" handoff: the planner's notes for the writer.
    std::string brief;
    /// The feeling of a sticker to send with the reply ("无语"); empty: none.
    std::string sticker;
    /// Style "sticker": only a sticker, no words.
    bool sticker_only = false;
    /// The wait tool: seconds to wait before looking again (0: not waiting).
    i64 wait = 0;
    /// Decided after she already spoke this turn.
    bool follow_up = false;
    std::string reason; ///< why she stays silent or waits
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
    /// The reply tool asks for `brief` (notes) instead of `intent`.
    bool reference_handoff = false;
    /// Her latest decisions, oldest first ("[21:05] 没说话：…"), so the planner
    /// knows what she just did and why.
    std::vector<std::string> decisions;
    /// Offer the search_history tool (long-term memory is enabled).
    bool memory_search = false;
    /// Offer stickers (the sticker pool is not empty).
    bool stickers = false;
    /// "- 名字：事实；事实" lines about the people in the new messages.
    std::string people;
    /// What she has said about herself before ("- [MM-DD] 她……").
    std::string self_notes;
    /// A summary of older chat the recent messages bring to mind.
    std::string recall;
    /// Meanings of slang others use in the log ("以下黑话来自…").
    std::string jargon;
};

/// The planner.md rule about search_history; empty when memory is off.
std::string_view memory_rule(bool enabled);

/// The planner.md rule about stickers; empty when she has none to send.
std::string_view sticker_rule(bool enabled);

/// The planner.md line describing what it hands the writer.
std::string_view handoff_rule(bool reference);

struct PlannerRequest {
    provider::History history;
    std::vector<provider::ToolDefinition> tools;
    /// ids[n - 1] is the message id shown as "#n".
    std::vector<i64> ids;
};

PlannerRequest build_planner_request(const PlannerInput &input);

/// Reads the planner's tool call. No call, an unknown tool or an invalid
/// target means silence; wait is clamped to 1..60 seconds (30 when missing).
Plan parse_plan(const std::vector<provider::OutputItem> &items, const std::vector<i64> &ids);

/// Messages that arrived during a turn, numbered on from the planner's
/// earlier lines ("#n [HH:MM] 名字(id): 内容"); appends their ids to `ids`.
std::string number_messages(const std::vector<store::StoredMessage> &messages, std::vector<i64> &ids, i32 utc_offset_minutes);

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
    /// Facts from the search step; the writer picks one point from them.
    std::string reference;
    /// Learned patterns the selector picked for this reply, "当X时，可以Y".
    std::vector<std::string> habits;
    /// Older chat the planner looked up for this reply (search_history).
    std::string remembered;
    /// Facts about the person she answers ("住在成都；在学吉他").
    std::string known;
    /// What she has said about herself before ("- [MM-DD] 她……"), to stay
    /// consistent with.
    std::string self_notes;
    /// What she already said to this same target, when she answers it again.
    std::string previous_reply;
    /// A random handful of phrases people in the groups actually say, to
    /// anchor her voice in real chat rather than the model's own register.
    std::vector<std::string> voice;
};

/// The quoted phrase of a learned style: 用"何意味" -> 何意味; empty if none.
std::string quoted_phrase(std::string_view style);

/// The search step: one call with only the web_search tool, asked to
/// summarise what it finds about the target message in a few sentences.
provider::History build_search_history(const ChatInfo &info, const store::StoredMessage &target, const Plan &plan, i64 now,
                                       i32 utc_offset_minutes);

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

/// The writer sometimes continues the transcript ("昵称: 内容") instead of
/// answering. Strips other speakers' labels and a bare "12345:" from reply
/// lines, and drops lines that only repeat a message in `recent`.
std::string strip_transcript(std::string_view text, const std::vector<store::StoredMessage> &recent);

} // namespace lilia::chat
