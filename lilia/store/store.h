#pragma once

#include <filesystem>
#include <string>
#include <string_view>
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

/// A learned expression pattern: 当 situation 时，可以 style.
struct Expression {
    i64 id = 0;
    std::string situation;
    std::string style;
    i64 count = 1; ///< times learned (from any group)
    i64 last_active = 0;
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

    /// True when lilia itself sent `message_id` in `chat`.
    Result<bool> is_own_message(const ChatKey &chat, i64 message_id);

    Result<void> add_llm_call(const LlmCall &call);

    /// Input plus output tokens of every call at or after `since` (unix seconds).
    Result<u64> tokens_since(i64 since);

    Result<void> add_media(const MediaRecord &media);

    /// Media rows of one message, in segment order.
    Result<std::vector<MediaRecord>> message_media(const ChatKey &chat, i64 message_id);

    /// A description already made for the same content; empty when none.
    Result<std::string> media_description(std::string_view sha1);

    /// Replaces the plain-text projection of a stored message (e.g. once its
    /// images have been described).
    Result<void> update_message_text(const ChatKey &chat, i64 message_id, std::string_view text);

    /// Adds a pattern, or counts it again when the style is already known
    /// (the newest situation wins).
    Result<void> learn_expression(std::string_view situation, std::string_view style, i64 now);

    /// Patterns, most counted first; `limit` 0 returns all of them.
    Result<std::vector<Expression>> expressions(usize limit);

    /// Up to `limit` patterns picked at random, excluding the `skip_top`
    /// that `expressions` would list first.
    Result<std::vector<Expression>> random_expressions(usize skip_top, usize limit);

    /// Keeps the `keep` most counted (then most recent) patterns.
    Result<void> prune_expressions(usize keep);

    Database database;
};

} // namespace lilia::store
