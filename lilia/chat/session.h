#pragma once

#include <deque>
#include <string>
#include <string_view>
#include <vector>

#include <lilia/chat/context.h>
#include <lilia/provider/common.h>
#include <lilia/provider/history.h>
#include <lilia/store/store.h>

namespace lilia::chat {

// The planner's history of one chat, the way MaiBot's planner keeps one: the
// chat's lines (stored rows after the window's anchor) interleaved with the
// planner's own rounds (its analysis, its tool call and the result). Each
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
    std::vector<std::string> facts; ///< "#18345 @ 了她" and the like
    std::string presence;           ///< just woke up, was busy, back online
    std::string self_notes;         ///< what she said about herself before
    std::string jargon;             ///< slang in the recent lines
    std::string recall;             ///< an older summary the chat brings back
    i64 now = 0;
    i32 utc_offset_minutes = 480;
};

/// "#18345 [21:05] 小明(20001): 内容": a line of the planner's history; the
/// number is the row id, stable for as long as the line is shown.
std::string session_line(const store::StoredMessage &message, i32 utc_offset_minutes);

/// The planner request: the system prompt, then `rows` and `records` merged
/// by row (a record closes the lines before it; its results open the next
/// user message, followed by the lines that came after), the last line ending
/// a cached prefix, then `tail` as a final part.
provider::History render_session(std::string_view prompt, const ChatInfo &info, const std::vector<store::StoredMessage> &rows,
                                 const std::deque<PlannerRecord> &records, const PlannerTail &tail);

/// After the window's anchor moved to `anchor_row`: records about lines no
/// longer shown go, and all but the newest `keep_cycles` turns lose their
/// analysis text (their tool calls and results stay).
void trim_records(std::deque<PlannerRecord> &records, i64 anchor_row, usize keep_cycles);

/// Which planner tools a chat offers.
struct PlannerToolSet {
    bool reference_handoff = false; ///< reply asks for `brief` instead of `intent`
    bool memory = false;            ///< search_history
    bool people = false;            ///< look_up_person
    bool web = false;               ///< search_web
    bool stickers = false;          ///< send_sticker, and reply's sticker
    bool poke = false;              ///< poke (groups only)
};

/// The planner's tools; the same every round, so they never break the cache.
std::vector<provider::ToolDefinition> planner_tools(const PlannerToolSet &set);

/// Reads the planner's tool call. Targets are "#<row id>" among `rows`; no
/// call, an unknown tool or an invalid target means silence. Waits are
/// clamped to 1..60 seconds (30 when missing).
Plan parse_plan(const std::vector<provider::OutputItem> &items, const std::vector<store::StoredMessage> &rows);

} // namespace lilia::chat
