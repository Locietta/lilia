#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <lilia/error.h>
#include <lilia/store/sqlite.h>

namespace lilia::store {

/// Conversation key. `type` is "group" or "private"; `id` is the group number
/// or the peer's QQ number.
struct ChatKey {
    std::string type;
    i64 id = 0;

    auto operator<=>(const ChatKey &) const = default;
};

struct StoredMessage {
    ChatKey chat;
    i64 message_id = 0;
    i64 sender_id = 0;
    std::string sender_name;
    i64 time = 0; ///< unix seconds
    std::string segments_json;
    std::string text; ///< onebot::render_plain projection
    bool from_self = false;
    i64 row_id = 0; ///< messages.id; filled by queries that need an order key
};

/// Which model calls a token sum covers; see Store::tokens_since.
enum struct TokenScope {
    ALL,
    CHATS,
    BACKGROUND,
};

struct LlmCall {
    i64 time = 0;
    ChatKey chat;
    std::string model;
    u64 input_tokens = 0;
    u64 output_tokens = 0;
    u64 cache_read_tokens = 0;
    u64 cache_write_tokens = 0;
    std::string stop;
    std::string request_id;
    std::string error; ///< empty on success
};

/// One archived (or failed) media segment of a stored message.
struct MediaRecord {
    ChatKey chat;
    i64 message_id = 0;
    i64 segment_index = 0;
    std::string kind;   ///< archive::to_string(MediaKind)
    std::string status; ///< "stored", "failed" or "too_large"
    std::string sha1;   ///< content hash; empty unless stored
    std::string path;   ///< relative to the data directory; empty unless stored
    i64 bytes = 0;
    std::string name;   ///< original file name, when known
    std::string source; ///< url or napcat path the bytes came from
    std::string error;
    i64 time = 0;
    /// What a vision model saw in an image or sticker; empty otherwise.
    std::string description;
};

/// A few consecutive messages of one chat, for memory search. `text` is
/// shown ("[10-05 21:03] 名字: 内容" lines); `search_text` is embedded and
/// keyword-indexed (the same lines without times).
struct MemoryChunk {
    i64 id = 0;
    ChatKey chat;
    i64 first_row = 0; ///< messages.id range covered
    i64 last_row = 0;
    i64 start_time = 0;
    i64 end_time = 0;
    std::string text;
    std::string search_text;
};

/// An incremental summary of a stretch of chat (empty `summary`: nothing
/// worth keeping, a checkpoint only). `cues` are newline-separated recall
/// cues.
struct ChatSummary {
    i64 id = 0;
    ChatKey chat;
    i64 first_row = 0;
    i64 last_row = 0;
    i64 start_time = 0;
    i64 end_time = 0;
    std::string summary;
    std::string cues;
};

/// A sticker seen in any group. `status`: "pending" (not described yet),
/// "registered" (may be sent), "spare" (fine but outside the pool) or
/// "rejected". `path` is relative to the data directory. `modified_by` is
/// "USER" once a person has edited the tags or description, which machine
/// tagging then leaves alone.
struct Sticker {
    i64 id = 0;
    std::string sha1;
    std::string path;
    std::string media_type;
    i64 bytes = 0;
    std::string description;
    std::string tags; ///< "无语、震惊"
    std::string status = "pending";
    i64 seen = 1;
    i64 used = 0;
    i64 first_seen = 0;
    i64 last_seen = 0;
    i64 last_used = 0;
    std::string modified_by = "AI";
    /// What people send it for, learned from the chats it turns up in
    /// (sticker_evidence); usage_count snippets so far, the last inference at
    /// usage_inferred_count. tags_by: "VISION" or "USAGE".
    std::string usage;
    i64 usage_count = 0;
    i64 usage_inferred_count = 0;
    std::string tags_by = "VISION";
};

/// An anonymised snippet of a chat a sticker turned up in; `own` when lilia
/// sent it.
struct StickerEvidence {
    i64 id = 0;
    i64 sticker_id = 0;
    ChatKey chat;
    std::string snippet;
    bool own = false;
    i64 time = 0;
};

/// A stable fact about a person, quoted from their own message. `fact` is
/// written about "TA".
struct PersonFact {
    i64 id = 0;
    i64 person_id = 0; ///< QQ number
    std::string fact;
    std::string quote;
    ChatKey chat;
    i64 message_id = 0;
    i64 time = 0;
};

/// A learned expression pattern: 当 situation 时，可以 style. `chat` is where
/// it was learned (empty type: before schema 9, shared). `situations` lists
/// every situation merged into it, newline-separated (MaiBot's content_list).
struct Expression {
    i64 id = 0;
    ChatKey chat;
    std::string situation;
    std::string style;
    std::string situations;
    i64 count = 1; ///< times learned
    i64 first_seen = 0;
    i64 last_active = 0;  ///< learned or selected
    bool checked = false; ///< approved by a person
    std::string modified_by = "AI";
};

/// One verdict of the learning review: "kept", "rejected" or "rescued".
struct ExpressionReview {
    i64 id = 0;
    i64 time = 0;
    ChatKey chat;
    std::string situation;
    std::string style;
    std::string verdict;
    std::string reason;
};

/// A slang term (MaiBot's jargon). `is_jargon`: -1 unknown yet, 0 ordinary
/// word, 1 jargon. Meanings are inferred at count thresholds; MANUAL entries
/// are never touched by inference.
struct Jargon {
    i64 id = 0;
    std::string content;
    std::string meaning;
    i64 is_jargon = -1;
    i64 count = 0;
    i64 last_inference_count = 0;
    bool complete = false;
    std::string created_by = "AI";
    i64 first_seen = 0;
    i64 last_seen = 0;
};

/// Context where a slang term was seen: the line and three on each side,
/// speakers anonymised.
struct JargonEvidence {
    i64 id = 0;
    ChatKey chat;
    std::string snippet;
    i64 time = 0;
};

/// Occurrences of a word or phrase in one batch of messages.
struct TermTally {
    std::string term;
    i64 count = 0;
    i64 message_count = 0; ///< messages it occurs in
};

/// A chat's high-frequency word or phrase (MaiBot's HighFrequencyTerm):
/// running totals, ranked from 1 by occurrences; frequencies are shares of
/// the chat's kept terms.
struct FrequentTerm {
    std::string term;
    i64 occurrence_count = 0;
    i64 message_count = 0;
    i64 rank = 0;
    double frequency = 0;
    double message_frequency = 0;
    i64 created_at = 0;
    i64 updated_at = 0;
};

/// Message times (unix seconds) of a group that learning has covered, live
/// or from fetched history. Gaps inside it (her downtime) are not tracked.
struct LearnedSpan {
    i64 first_time = 0;
    i64 last_time = 0;
};

struct Store {
    static Result<Store> open(const std::filesystem::path &path);
    /// Existing database without schema changes; for inspection tools.
    static Result<Store> open_readonly(const std::filesystem::path &path);
    static Result<Store> open_memory();

    /// Duplicates (same chat and message id) are ignored.
    Result<void> add_message(const StoredMessage &message);

    /// The newest `limit` messages of a chat, oldest first.
    Result<std::vector<StoredMessage>> recent_messages(const ChatKey &chat, usize limit);

    /// One stored message by its QQ id (row_id set); nullopt when not stored.
    Result<std::optional<StoredMessage>> message(const ChatKey &chat, i64 message_id);

    /// The name `user_id` had on their newest stored message in `chat`;
    /// empty when they have none.
    Result<std::string> sender_name(const ChatKey &chat, i64 user_id);

    Result<void> add_llm_call(const LlmCall &call);

    /// Input plus output tokens of the calls at or after `since` (unix
    /// seconds): all of them, those made for chats (group and private), or
    /// the background ones (learning, sticker tagging).
    Result<u64> tokens_since(i64 since, TokenScope scope = TokenScope::ALL);

    Result<void> add_media(const MediaRecord &media);

    /// Media rows of one message, in segment order.
    Result<std::vector<MediaRecord>> message_media(const ChatKey &chat, i64 message_id);

    /// A description already made for the same content; empty when none.
    Result<std::string> media_description(std::string_view sha1);

    /// Replaces the plain-text projection of a stored message (e.g. once its
    /// images have been described).
    Result<void> update_message_text(const ChatKey &chat, i64 message_id, std::string_view text);

    /// Adds a pattern learned in `chat`, or counts it again when that chat
    /// already has the style (the newest situation leads; all are kept).
    /// Returns its id.
    Result<i64> learn_expression(const ChatKey &chat, std::string_view situation, std::string_view style, i64 now);

    /// Marks patterns as selected now.
    Result<void> touch_expressions(const std::vector<i64> &ids, i64 now);

    /// A person approves (or withdraws approval of) a pattern.
    Result<void> set_expression_checked(i64 id, bool checked);

    Result<void> add_expression_review(const ExpressionReview &review);

    /// Review verdicts, newest first; `verdict` empty lists all.
    Result<std::vector<ExpressionReview>> expression_reviews(std::string_view verdict, usize limit);

    Result<std::optional<ExpressionReview>> expression_review(i64 id);

    /// All slang terms, most seen first.
    Result<std::vector<Jargon>> jargon_terms();

    Result<std::optional<Jargon>> jargon_by_content(std::string_view content);

    /// Counts a sighting of `content` in `chat` with its evidence (the newest
    /// `keep_evidence` are kept per term); creates the term. Returns it.
    Result<Jargon> add_jargon_hit(std::string_view content, const ChatKey &chat, std::string_view snippet, i64 now, usize keep_evidence);

    Result<std::vector<JargonEvidence>> jargon_evidence(i64 jargon_id, usize limit);

    /// Chats (ids) a term has been seen in.
    Result<std::vector<i64>> jargon_chats(i64 jargon_id);

    /// Records an inference: meaning, verdict and the count it was made at.
    Result<void> set_jargon_inference(i64 id, std::string_view meaning, i64 is_jargon, i64 at_count, bool complete);

    /// A person defines (or redefines) a term; inference leaves it alone.
    Result<void> set_jargon_manual(std::string_view content, std::string_view meaning, i64 now);

    Result<void> delete_jargon(std::string_view content);

    /// Adds a batch's tallies to `chat`'s high-frequency terms, keeps the top
    /// `max_terms` (by occurrences, then messages, longer first, then text)
    /// and re-ranks them.
    Result<void> merge_frequent_terms(const ChatKey &chat, const std::vector<TermTally> &tallies, i64 now, usize max_terms);

    /// A chat's high-frequency terms, by rank.
    Result<std::vector<FrequentTerm>> frequent_terms(const ChatKey &chat);

    /// Patterns, most counted first; `limit` 0 returns all of them.
    Result<std::vector<Expression>> expressions(usize limit);

    /// The style text of every learned expression (no vectors), in no order.
    Result<std::vector<std::string>> expression_styles();

    /// Messages of `chat` stored after row `after_row`, oldest first (with
    /// row_id set), at most `limit`.
    Result<std::vector<StoredMessage>> messages_after(const ChatKey &chat, i64 after_row, usize limit);

    Result<void> add_chunk(const MemoryChunk &chunk);

    /// The last message row covered by chunks of `chat` (0: none).
    Result<i64> last_chunk_row(const ChatKey &chat);

    /// All chunks of `chat`, oldest first.
    Result<std::vector<MemoryChunk>> chunks(const ChatKey &chat);

    /// The search text of every chunk of every chat (for the embedding backfill).
    Result<std::vector<std::string>> search_texts();

    Result<void> add_summary(const ChatSummary &summary);

    /// The last message row covered by summaries of `chat` (0: none).
    Result<i64> last_summary_row(const ChatKey &chat);

    /// Summaries of `chat`, oldest first; `limit` > 0 keeps only the newest.
    Result<std::vector<ChatSummary>> summaries(const ChatKey &chat, usize limit);

    /// Every summary of every chat.
    Result<std::vector<ChatSummary>> all_summaries();

    /// The sticker with this content hash, if known.
    Result<std::optional<Sticker>> sticker_by_hash(std::string_view sha1);
    Result<std::optional<Sticker>> sticker_by_id(i64 id);
    Result<i64> add_sticker(const Sticker &sticker);
    /// Counts another sighting.
    Result<void> sticker_seen(i64 id, i64 time);
    /// Machine tags and the status; the tags and description of a sticker a
    /// person edited stay as they are.
    Result<void> describe_sticker(i64 id, std::string_view description, std::string_view tags, std::string_view status);
    /// A person's tags and description, kept from then on.
    Result<void> edit_sticker(i64 id, std::string_view tags, std::string_view description);
    /// Adds a sighting's snippet (only the newest `keep` stay) and counts it;
    /// returns the sticker as it is now.
    Result<Sticker> add_sticker_evidence(i64 id, const ChatKey &chat, std::string_view snippet, bool own, i64 time, usize keep);
    /// The newest `limit` snippets of a sticker, oldest first.
    Result<std::vector<StickerEvidence>> sticker_evidence(i64 id, usize limit);
    /// The usage inferred at `at_count` snippets (nullopt: unchanged), and its
    /// tags unless a person edited them.
    Result<void> set_sticker_usage(i64 id, std::optional<std::string_view> usage, std::optional<std::string_view> tags, i64 at_count);
    /// Forgets a group's snippets; returns how many went.
    Result<i64> delete_sticker_evidence_of_group(i64 chat_id);
    Result<void> set_sticker_status(i64 id, std::string_view status);
    Result<void> sticker_used(i64 id, i64 time);
    /// Stickers with this status, most seen first.
    Result<std::vector<Sticker>> stickers(std::string_view status);

    Result<void> add_fact(const PersonFact &fact);

    /// Facts about `person_id`, newest first; `limit` 0 returns all.
    Result<std::vector<PersonFact>> facts(i64 person_id, usize limit);

    /// Every fact about everyone.
    Result<std::vector<PersonFact>> all_facts();

    Result<void> delete_fact(i64 id);

    /// Stored vectors of `profile` for these text hashes: (hash, raw
    /// little-endian float32) for each one found.
    Result<std::vector<std::pair<std::string, std::string>>> embeddings(std::string_view profile, const std::vector<std::string> &hashes);

    /// Stores (hash, vector) rows under `profile`, in one transaction.
    Result<void> add_embeddings(std::string_view profile, const std::vector<std::pair<std::string, std::string>> &rows);

    /// Drops vectors of every other profile (another model); returns how many.
    Result<i64> delete_embeddings_except(std::string_view profile);

    /// The stretch of `group_id` already learned from; nullopt when none.
    Result<std::optional<LearnedSpan>> learned_span(i64 group_id);
    /// Widens the span to cover [first_time, last_time].
    Result<void> extend_learned_span(i64 group_id, i64 first_time, i64 last_time);

    /// Deletes the patterns with these ids.
    Result<void> delete_expressions(const std::vector<i64> &ids);

    /// Keeps the `keep` most counted (then most recent) patterns.
    Result<void> prune_expressions(usize keep);

    Database database;
};

} // namespace lilia::store
