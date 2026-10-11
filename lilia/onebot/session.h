#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <lighter/async/runtime/task.h>

#include <lilia/net/websocket.h>
#include <lilia/onebot/types.h>

namespace lilia::onebot {

enum struct OneBotErrorKind {
    TRANSPORT,     ///< websocket failure; see `ws`
    DECODE,        ///< a frame was not valid OneBot JSON
    TIMEOUT,       ///< no response (or no traffic) within the budget
    DISCONNECTED,  ///< the session ended while waiting
    ACTION_FAILED, ///< the implementation answered with a failure status
};

struct OneBotError {
    OneBotErrorKind kind = OneBotErrorKind::TRANSPORT;
    std::string detail;
    i64 retcode = 0;
    std::optional<net::WsError> ws;

    std::string message() const;
};

struct SessionOptions {
    net::WsConnectOptions connect;
    /// Sent as `Authorization: Bearer`; empty sends no header.
    std::string access_token;
    std::chrono::milliseconds action_timeout{30'000};
    /// Treat the connection as dead after this long without any frame.
    /// NapCat sends a heartbeat every 30s by default.
    std::chrono::milliseconds idle_timeout{95'000};
};

using EventHandler = std::move_only_function<void(Event)>;

/// One live OneBot 11 forward-websocket connection. Shared ownership lets
/// in-flight actions outlive the reader loop that serves them.
struct Session : std::enable_shared_from_this<Session> {
    static lighter::Task<std::shared_ptr<Session>, OneBotError> open(SessionOptions options);

    /// Reads frames until the connection ends, routing action responses to
    /// their callers and every event to `on_event`. Always ends with the error
    /// that closed the session; pending actions then fail with DISCONNECTED.
    lighter::Task<void, OneBotError> run(EventHandler on_event);

    /// Calls an action with already-encoded JSON params and returns the raw
    /// JSON `data` of a successful response.
    lighter::Task<std::string, OneBotError> call(std::string action, std::string params_json);

    lighter::Task<LoginInfo, OneBotError> get_login_info();
    lighter::Task<SentMessage, OneBotError> send_group_msg(i64 group_id, Message message);
    lighter::Task<SentMessage, OneBotError> send_private_msg(i64 user_id, Message message);

    void close() noexcept;
    bool is_open() const noexcept;

    struct State;
    std::unique_ptr<State> state;

    Session();
    ~Session();
};

/// Up to `count` of a group's newest messages, oldest first, paged through
/// get_group_msg_history; fewer when NapCat fails (logged).
lighter::Task<std::vector<MessageEvent>> group_history(std::shared_ptr<Session> session, i64 group, usize count);

/// Classifies one incoming frame. Exposed for tests.
struct Frame {
    std::optional<std::string> echo;
    std::optional<std::string> post_type;
};

lighter::codec::json::Result<Event> decode_event(std::string_view text, std::string_view post_type);

} // namespace lilia::onebot
