#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/async/runtime/task.h>

#include <lilia/chat/context.h>
#include <lilia/chat/responder.h>
#include <lilia/error.h>
#include <lilia/onebot/message.h>
#include <lilia/provider/common.h>
#include <lilia/provider/history.h>
#include <lilia/store/store.h>

// What the parts of the responder share: responder.cpp (connections,
// ingest, the turn worker), turn.cpp (the planner loop), reply.cpp (the
// writer and sending a reply), notices.cpp (notice lines) and calls.cpp
// (model calls and sending).

namespace lilia::chat::detail {

/// A prompt compiled into the binary.
inline std::string_view embedded(std::span<const std::byte> content) {
    return {reinterpret_cast<const char *>(content.data()), content.size()};
}

/// "group:123" for logs.
std::string describe(const store::ChatKey &chat);

/// Sends one message through the outbox or the live session; its id, or
/// nullopt when it could not be sent.
lighter::Task<std::optional<i64>> send_message(Responder *r, const store::ChatKey &chat, onebot::Message message);

/// Sends one message and records it as lilia's own (as `text` when given,
/// else as rendered); its id, or nullopt when it could not be sent.
lighter::Task<std::optional<i64>> send_and_record(Responder *r, const store::ChatKey &chat, onebot::Message message, std::string text = {});

/// One OneBot action through the live session; false when it is gone or the
/// call failed.
lighter::Task<bool> call_action(Responder *r, std::string action, std::string params);

/// Runs one model call, records its usage, and returns the completed output
/// items (nullopt on failure, with the error in `failure` when given).
lighter::Task<std::optional<std::vector<provider::OutputItem>>>
call_model(Responder *r, provider::Provider &provider, std::string model, std::string_view purpose, const store::ChatKey &chat,
           const provider::History &history, const std::vector<provider::ToolDefinition> &tools, Error *failure = nullptr);

/// A planner-model call, then each planner fallback in turn while they fail.
lighter::Task<std::optional<std::vector<provider::OutputItem>>> call_planner(Responder *r, std::string_view purpose,
                                                                             const store::ChatKey &chat, const provider::History &history,
                                                                             const std::vector<provider::ToolDefinition> &tools);

/// Text after the last non-text item; earlier text is narration such as
/// "let me search for that".
std::string final_text(const std::vector<provider::OutputItem> &items);

/// A person in a notice line: her name in the chat, the name on their newest
/// message, or their QQ number.
std::string person(Responder *r, const store::ChatKey &chat, i64 user);

/// Who did what a notice line tells (a poke's poker); 0 when unknown.
i64 notice_actor(const store::StoredMessage &line);

/// She has stickers she may send.
bool stickers_ready(Responder *r);

/// The planner's find_sticker: the registered stickers closest to `feeling`
/// (by embedding, or by words without one), at most `stickers.candidates`.
lighter::Task<std::vector<store::Sticker>> find_stickers(Responder *r, std::string feeling);

/// Sends `sticker` as a QQ sticker and records it ("[表情: 描述]"); false
/// when it could not be read or sent.
lighter::Task<bool> send_sticker(Responder *r, store::ChatKey chat, store::Sticker sticker);

/// What one reply came to.
struct Delivered {
    bool sent = false;
    std::string said;             ///< her messages, joined
    std::vector<i64> message_ids; ///< of her messages, in order
    /// Why nothing was attempted (the target was recalled), for the planner.
    std::string refused;
};

/// One reply as the planner asked for it: a sticker alone, or the writer's
/// text (with what the planner looked up this turn and the picked
/// expressions) sent the way MaiBot sends replies, plus a sticker when asked.
/// Records what she said.
lighter::Task<Delivered> deliver(Responder *r, store::ChatKey chat, Plan plan, std::vector<store::StoredMessage> log_lines,
                                 store::StoredMessage target, ChatInfo info, i64 started, std::string remembered, std::string reference,
                                 std::string self_notes, std::string presence);

} // namespace lilia::chat::detail
