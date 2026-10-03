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

struct Store {
    static Result<Store> open(const std::filesystem::path &path);
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

    Database database;
};

} // namespace lilia::store
