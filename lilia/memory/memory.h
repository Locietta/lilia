#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/async/runtime/task.h>

#include <lilia/config.h>
#include <lilia/provider/history.h>
#include <lilia/provider/provider.h>
#include <lilia/error.h>
#include <lilia/retrieval/retriever.h>
#include <lilia/store/store.h>

namespace lilia::memory {

// Long-term memory, part one: searchable history of the allowed chats.
// Consecutive messages are grouped into small chunks; a search ranks a
// chat's chunks and summaries by meaning and by keywords (retrieval/), the
// newer a little higher.

struct Hit {
    store::MemoryChunk chunk;
    double score = 0; ///< similarity to the query
};

/// Splits messages (oldest first) into chunks of at most `size`, starting a
/// new chunk at gaps of `gap_seconds`. The last group is complete only when
/// it is full or older than `gap_seconds` at `now`; incomplete groups are
/// left for later and not returned.
std::vector<std::vector<store::StoredMessage>> complete_chunks(const std::vector<store::StoredMessage> &messages, usize size,
                                                               i64 gap_seconds, i64 now);

/// "[10-05 21:03] 名字: 内容" lines, the stored and displayed form.
std::string display_text(const std::vector<store::StoredMessage> &messages, std::string_view self_name, i32 utc_offset_minutes);

/// "名字: 内容" lines without times, the embedded form.
std::string embed_text(const std::vector<store::StoredMessage> &messages, std::string_view self_name);

/// Search results as the planner and the writer see them.
std::string render_hits(const std::vector<Hit> &hits);

// Facts about people (MaiBot's person-fact writeback): extracted after she
// answers someone, only from that person's own words, each with a verbatim
// quote that is checked here; written about "TA" and keyed by QQ number.

struct FactRequest {
    provider::History history;
    std::vector<provider::ToolDefinition> tools;
};

struct ExtractedFact {
    std::string fact;
    std::string quote;
    usize source = 0; ///< 0-based index into the person's own messages
};

/// The person's own messages are numbered; the nearby chat is context only.
FactRequest build_fact_request(std::string_view prompt, std::string_view person, const std::vector<store::StoredMessage> &own,
                               const std::vector<store::StoredMessage> &context, std::string_view self_name);

/// Reads record_facts, keeping facts whose quote really is in the cited
/// message (spaces ignored) and whose length is sane; at most 5.
std::vector<ExtractedFact> parse_facts(const std::vector<provider::OutputItem> &items, const std::vector<store::StoredMessage> &own);

/// "对芒果过敏；住在杭州": facts about TA, newest first, joined.
std::string render_facts(const std::vector<store::PersonFact> &facts);

// Chat summaries, two kinds. Rolling summaries (MaiBot's incremental chat
// summary): every `summary_messages` messages of a chat get one summary of
// what is new and worth keeping, seeing the previous ones; they are
// searchable. Window summaries (MaiBot's mid-term memory): what the planner's
// log window drops is summarised whole, with recall cues; the newest few lead
// the planner's log, and an older one comes back when its cues match the chat.

/// Window summaries shown ahead of the planner's log. MaiBot keeps the newest
/// 10 in its history too, but hides them from the planner and only recalls.
inline constexpr usize k_earlier_summaries = 10;

struct SummaryResult {
    std::string summary;
    std::vector<std::string> cues;
};

/// `previous` (oldest first) are shown for context only.
FactRequest build_summary_request(std::string_view prompt, const std::vector<store::ChatSummary> &previous, std::string_view lines);

/// Reads record_summary; nullopt when the tool was not called. An empty
/// summary is a valid "nothing new" answer.
std::optional<SummaryResult> parse_summary(const std::vector<provider::OutputItem> &items);

/// "[10-05 21:00 至 22:30 的聊天摘要] ..." as a rankable chunk.
store::MemoryChunk summary_chunk(const store::ChatSummary &summary, i32 utc_offset_minutes);

/// What a summary is found by: the summary, then each recall cue.
std::vector<std::string> summary_texts(const store::ChatSummary &summary);

/// Everything memory embeds (chunks, summaries and their cues, facts), for
/// the startup backfill.
std::vector<std::string> embedded_texts(store::Store &store);

struct Memory {
    Memory(const MemoryConfig &config, std::string self_name, i32 utc_offset_minutes, store::Store &store, retrieval::Retriever &retriever);

    /// Chunks and embeds the messages of `chat` not covered yet (the whole
    /// history the first time). Calls for a chat while one runs are merged.
    lighter::Task<> index(store::ChatKey chat);

    /// The best chunks of `chat` for `query`.
    lighter::Task<std::vector<Hit>> search(store::ChatKey chat, std::string query);

    /// Stores `fact` unless the person already has one meaning the same
    /// (same text, or cosine >= the duplicate threshold). True if stored.
    /// `replace`: a close match is replaced instead (her own facts change:
    /// "还没看" becomes "看完了").
    lighter::Task<bool> remember(store::PersonFact fact, bool replace = false);

    /// What she has said about herself (facts stored under `self_id`) that
    /// bears on `context`: the `limit` closest plus the newest few, oldest
    /// first, each "- [MM-DD] 她……"; empty when there is none.
    lighter::Task<std::string> self_notes(i64 self_id, std::string context, usize limit);

    /// Summarizes the next `summary_messages` messages of `chat` if that many
    /// are waiting (called by index()). Needs `summarizer`.
    lighter::Task<> summarize(store::ChatKey chat);

    /// After the planner's window of `chat` moved past `through_row`: one
    /// window summary of the lines up to it that no window summary covers
    /// yet, at most the newest `span`. Skipped while one runs for the chat
    /// (the next covers its lines). Needs `summarizer`.
    lighter::Task<> summarize_window(store::ChatKey chat, i64 through_row, usize span);

    /// The newest `k_earlier_summaries` window summaries, oldest first, one
    /// "[MM-DD HH:MM 至 MM-DD HH:MM] summary" line each; empty when none.
    std::string earlier(const store::ChatKey &chat);

    /// A window summary whose cues match the last few messages of `log` well
    /// enough, among the older ones earlier() leaves out; empty when none does.
    lighter::Task<std::string> recall(store::ChatKey chat, std::vector<store::StoredMessage> log);

    std::move_only_function<i64()> now;
    const MemoryConfig &config;
    std::string self_name;
    i32 utc_offset_minutes;
    store::Store &store;
    retrieval::Retriever &retriever;
    /// Writes summaries; null leaves summarizing off.
    std::optional<provider::Provider> summarizer;
    std::string summarizer_model;
    std::vector<store::ChatKey> indexing;  ///< chats with an index() running
    std::vector<store::ChatKey> again;     ///< asked again while running
    std::vector<store::ChatKey> windowing; ///< chats with a summarize_window() running

private:
    /// One summary call over `messages` (oldest first, at least one), stored
    /// as `kind` and embedded; false when the call failed.
    lighter::Task<bool> write_summary(store::ChatKey chat, std::string_view kind, std::string_view prompt,
                                      std::vector<store::ChatSummary> previous, std::vector<store::StoredMessage> messages);

    /// Window summaries of `chat` with something in them, oldest first.
    std::vector<store::ChatSummary> window_summaries(const store::ChatKey &chat);
};

} // namespace lilia::memory
