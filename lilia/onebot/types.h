#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <glaze/json.hpp>

#include <lighter/codec/json/json.h>
#include <lighter/types.hpp>

namespace lilia::onebot {

using namespace lighter::types;

// OneBot 11 wire vocabulary as NapCat sends it with
// `messagePostFormat: "array"`. Segments stay generic so unknown segment
// types survive a round trip; typed helpers live in message.h.

/// NapCat adds fields freely; never fail on keys we do not model.
inline constexpr lighter::codec::json::Opts k_read_opts{{.null_terminated = false, .error_on_unknown_keys = false}};

/// Segment data keeps integers exact (generic_i64) because ids such as
/// message and group numbers travel through it.
struct Segment {
    std::string type;
    glz::generic_i64 data;
};

using Message = std::vector<Segment>;

struct Sender {
    i64 user_id = 0;
    std::string nickname;
    std::string card;
    std::string role;
};

struct MessageEvent {
    i64 time = 0;
    i64 self_id = 0;
    std::string message_type; ///< "group" or "private"
    std::string sub_type;
    i64 message_id = 0;
    i64 user_id = 0;
    std::optional<i64> group_id;
    std::optional<std::string> group_name;
    Sender sender;
    Message message;
    std::string raw_message;

    bool is_group() const noexcept { return message_type == "group" && group_id.has_value(); }
};

struct MetaEvent {
    i64 time = 0;
    i64 self_id = 0;
    std::string meta_event_type; ///< "lifecycle" or "heartbeat"
    std::string sub_type;
    std::optional<i64> interval;
};

/// A notice, e.g. a poke: notice_type "notify", sub_type "poke", `user_id`
/// poked `target_id`.
struct NoticeEvent {
    i64 time = 0;
    i64 self_id = 0;
    std::string notice_type;
    std::string sub_type;
    std::optional<i64> group_id;
    i64 user_id = 0;
    std::optional<i64> target_id;
};

/// Requests and anything else lilia does not handle yet.
struct OtherEvent {
    std::string post_type;
};

using Event = std::variant<MessageEvent, MetaEvent, NoticeEvent, OtherEvent>;

struct LoginInfo {
    i64 user_id = 0;
    std::string nickname;
};

struct SentMessage {
    i64 message_id = 0;
};

} // namespace lilia::onebot
