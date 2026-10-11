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
    std::string intent;
    /// "reference" handoff: the planner's notes for the writer.
    std::string brief;
    /// The send_sticker tool: the sticker to send ("s12").
    std::string sticker;
    /// The wait tool: seconds to wait before looking again (0: not waiting).
    i64 wait = 0;
    /// Decided after she already spoke this turn.
    bool follow_up = false;
    /// The repeat tool: send `target`'s text again as it is (复读).
    bool repeat = false;
    /// The poke tool: poke whoever wrote `target` (or poked her, for a poke notice).
    bool poke = false;
    /// The look_up_person tool: what she knows about whoever wrote `target`.
    bool look_up = false;
    std::string reason; ///< why she stays silent or waits
    /// What the planner wrote before its tool call (its view of the chat).
    std::string analysis;
};

/// Where the chat is ("这是 QQ 群「…」。她在这里的名字是…"); stable, so it
/// may lead a cached prefix.
std::string chat_place(const ChatInfo &info);

/// "现在是 2026-10-03 21:05（UTC+8）。": changes every call, so it goes last.
std::string clock_line(i64 now, i32 utc_offset_minutes);

/// The planner.md rules about search_history and look_up_person; empty when
/// both are off.
std::string memory_rule(bool history, bool people);

/// The planner.md rule about web_search; empty when it is off.
std::string_view search_rule(bool enabled);

/// The planner.md rule about stickers; empty when she has none to send.
std::string_view sticker_rule(bool enabled);

/// The planner.md line describing what it hands the writer.
std::string_view handoff_rule(bool reference);

struct WriterInput {
    std::string persona;
    std::string rules; ///< writer.md
    ChatInfo info;
    std::vector<store::StoredMessage> recent;
    store::StoredMessage target;
    /// Its analysis reaches the writer too, "#row" numbers replaced by the
    /// lines they stand for.
    Plan plan;
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
    /// Why she is only now answering (just woke up, was busy); empty usually.
    std::string presence;
    /// What she already said to this same target, when she answers it again.
    std::string previous_reply;
};

/// The planner's web_search tool: one call with only the provider-hosted
/// web_search tool, asked to summarise what it finds about `query`.
provider::History build_search_history(std::string_view query, i64 now, i32 utc_offset_minutes);

/// SYSTEM persona and rules; then the chat log as alternating turns (lilia's
/// own lines are assistant turns; the last one ends a cached prefix); then a
/// final user part with where and when, the style and notes, ending with a
/// note that it is her turn to answer `target`.
provider::History build_writer_history(const WriterInput &input);

/// "[HH:MM] name(qq): text", with "(她自己)" marking lilia's own lines (the
/// planner is told it is not her); group notices as "[HH:MM] （群通知）text".
std::string render_line(const store::StoredMessage &message, i32 utc_offset_minutes);

/// Sender name of the group notice lines lilia stores (sender_id 0).
inline constexpr std::string_view k_notice_sender = "群通知";

/// A stored group notice ("小明戳了戳莉莉娅"), not someone's message.
inline bool is_notice(const store::StoredMessage &message) { return !message.from_self && message.sender_id == 0; }

/// "[回复 小明: 前二十个字…] " for a message quoting `quoted`, which is named
/// `self_name` when it is hers; "[回复] " when the quoted message is unknown.
std::string quote_prefix(const store::StoredMessage *quoted, std::string_view self_name);

/// A stored text without its quote prefix: only what the sender wrote.
std::string_view without_quote(std::string_view text);

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
