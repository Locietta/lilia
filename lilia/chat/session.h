#pragma once

#include <deque>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include <lilia/chat/addressee.h>
#include <lilia/chat/context.h>
#include <lilia/provider/common.h>
#include <lilia/provider/history.h>
#include <lilia/store/store.h>

namespace lilia::chat {

// The planner's history of one chat, the way MaiBot's planner keeps one: the
// chat's lines (stored rows after the window's anchor) interleaved with the
// planner's own rounds (its analysis, its tool calls and their results). Each
// round renders it again; between trims it only grows, so consecutive calls
// share a cacheable prefix and the planner always sees what it thought and did
// before.

/// One planner round as it stays in the chat's history.
struct PlannerRecord {
    i64 after_row = 0; ///< the newest row the round's request showed
    u64 cycle = 0;     ///< the turn it belongs to
    /// Its analysis text and tool call(s); the text goes when the record is
    /// folded at a trim.
    std::vector<provider::OutputItem> output;
    /// One result per tool call, in call order.
    std::vector<ToolOutcome> outcomes;
};

/// What the planner is told per round and never kept: the newest lines,
/// notes, facts about the new messages, the time.
struct PlannerTail {
    i64 first_new_row = 0;          ///< 0: nothing new since it last looked
    std::vector<std::string> facts; ///< "#18345 提到了她的名字", a poke, 复读
    std::string hint;               ///< the addressee prior for new lines (feedback/)
    std::string presence;           ///< just woke up, was busy, back online
    std::string self_notes;         ///< what she said about herself before
    std::string jargon;             ///< slang in the recent lines
    std::string recall;             ///< an older window summary the chat brings back
    i64 now = 0;
    i32 utc_offset_minutes = 480;
};

/// "#18345 [21:05] 小明(20001)→小红: [回复 #18340 小红: …] 内容": a line of the
/// planner's history, with whom it is explicitly said to (`edges`, when
/// known). The number is the row id, stable for as long as the line is shown.
std::string session_line(const store::StoredMessage &message, i32 utc_offset_minutes, const LineEdges *edges = nullptr);

/// The planner request: the system prompt, then `rows` and `records` merged
/// by row (a record closes the lines before it; its results open the next
/// user message, followed by the lines that came after), the last line ending
/// a cached prefix, then `tail` as a final part. `edges` are the rows'
/// (aligned with them; empty: none shown). `earlier`, the summaries of what
/// scrolled out of the window (memory::render_earlier), leads the log.
provider::History render_session(std::string_view prompt, const ChatInfo &info, const std::vector<store::StoredMessage> &rows,
                                 const std::deque<PlannerRecord> &records, const PlannerTail &tail,
                                 const std::vector<LineEdges> &edges = {}, std::string_view earlier = {});

/// After the window's anchor moved to `anchor_row`: records about lines no
/// longer shown go, and all but the newest `keep_cycles` turns lose their
/// analysis text (their tool calls and results stay).
void trim_records(std::deque<PlannerRecord> &records, i64 anchor_row, usize keep_cycles);

/// Which planner tools a chat offers.
struct PlannerToolSet {
    bool memory = false;   ///< search_history
    bool people = false;   ///< look_up_person
    bool web = false;      ///< search_web
    bool stickers = false; ///< find_sticker and send_sticker
    bool poke = false;     ///< poke (groups only)
};

/// The planner's tools; the same every round, so they never break the cache.
std::vector<provider::ToolDefinition> planner_tools(const PlannerToolSet &set);

/// Reads the arguments of one planner tool call. Targets are "#<row id>"
/// among `rows`; waits are clamped to 1..60 seconds. A call it cannot read (a
/// target not shown, a wait without seconds) gives the receipt telling the
/// planner why. A tool with no arguments for a Plan (search_web, or a name that
/// is no tool) reads as an empty Plan.
std::expected<Plan, std::string> parse_call(const provider::ToolCall &call, const std::vector<store::StoredMessage> &rows);

/// parse_call() of the first tool call in `items` (a call it cannot read has
/// its receipt as the reason); no call means silence.
Plan parse_plan(const std::vector<provider::OutputItem> &items, const std::vector<store::StoredMessage> &rows);

} // namespace lilia::chat
