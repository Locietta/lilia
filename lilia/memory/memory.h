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

// Rolling chat summaries (MaiBot's incremental chat summary and mid-term
// recall): every `summary_messages` messages of a chat get one summary of
// what is new, seeing the previous ones, plus recall cues. Summaries are
// searchable; their cues bring them back when the chat returns to a topic
// that has scrolled out of view.

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

    /// A summary whose cues match the last few messages of `log` well enough,
    /// among those that end before `log` begins; empty when none does.
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
    std::vector<store::ChatKey> indexing; ///< chats with an index() running
    std::vector<store::ChatKey> again;    ///< asked again while running
};

} // namespace lilia::memory
