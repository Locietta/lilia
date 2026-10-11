#include "session.h"

#include <algorithm>
#include <format>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include <lighter/async/runtime/sync.h>
#include <lighter/async/runtime/timeout.h>

#include <lilia/log.h>

namespace lilia::onebot {

using lighter::outcome_error;
using lighter::Task;

namespace {

struct ActionResponse {
    std::string status;
    i64 retcode = 0;
    glz::raw_json data;
    std::optional<std::string> message;
    std::optional<std::string> wording;
};

struct SendGroupParams {
    i64 group_id = 0;
    Message message;
};

struct SendPrivateParams {
    i64 user_id = 0;
    Message message;
};

struct Pending {
    lighter::Event done;
    std::optional<lighter::Outcome<std::string, OneBotError>> result;
};

OneBotError transport(net::WsError error) {
    return {.kind = OneBotErrorKind::TRANSPORT, .detail = error.message(), .ws = std::move(error)};
}

OneBotError decode_error(std::string detail) { return {.kind = OneBotErrorKind::DECODE, .detail = std::move(detail)}; }

std::string json_string(std::string_view text) {
    auto encoded = glz::write_json(text);
    return encoded ? *std::move(encoded) : std::string("\"\"");
}

template <typename T>
lighter::Outcome<T, OneBotError> parse_data(std::string_view text, std::string_view what) {
    T value{};
    if (auto ec = glz::read<k_read_opts>(value, text)) {
        return outcome_error(decode_error(std::string(what) + ": " + glz::format_error(ec, text)));
    }
    return value;
}

} // namespace

std::string OneBotError::message() const {
    std::string out;
    switch (kind) {
        case OneBotErrorKind::TRANSPORT: out = "onebot transport error: "; break;
        case OneBotErrorKind::DECODE: out = "onebot decode error: "; break;
        case OneBotErrorKind::TIMEOUT: out = "onebot timeout: "; break;
        case OneBotErrorKind::DISCONNECTED: out = "onebot disconnected: "; break;
        case OneBotErrorKind::ACTION_FAILED: out = "onebot action failed (retcode " + std::to_string(retcode) + "): "; break;
    }
    out += detail;
    return out;
}

struct Session::State {
    net::WebSocket socket;
    SessionOptions options;
    std::map<std::string, std::shared_ptr<Pending>, std::less<>> pending;
    u64 next_echo = 0;
    bool open = true;
};

Session::Session() = default;
Session::~Session() = default;

lighter::codec::json::Result<Event> decode_event(std::string_view text, std::string_view post_type) {
    const auto read = [text]<typename T>(T value) -> lighter::codec::json::Result<Event> {
        if (auto ec = glz::read<k_read_opts>(value, text)) {
            return outcome_error(lighter::codec::json::Error{.code = ec.ec, .detail = glz::format_error(ec, text)});
        }
        return Event(std::move(value));
    };
    if (post_type == "message") return read(MessageEvent{});
    if (post_type == "meta_event") return read(MetaEvent{});
    if (post_type == "notice") return read(NoticeEvent{});
    return Event(OtherEvent{.post_type = std::string(post_type)});
}

namespace {

void settle(Pending &pending, lighter::Outcome<std::string, OneBotError> result) {
    if (pending.result) return;
    pending.result.emplace(std::move(result));
    pending.done.set();
}

void dispatch(Session::State &state, std::string_view text, EventHandler &on_event) {
    Frame frame;
    if (auto ec = glz::read<k_read_opts>(frame, text)) {
        log::warn("onebot: ignoring undecodable frame: {}", glz::format_error(ec, text));
        return;
    }

    if (frame.echo) {
        const auto it = state.pending.find(*frame.echo);
        if (it == state.pending.end()) {
            log::debug("onebot: response for unknown echo {}", *frame.echo);
            return;
        }
        auto pending = it->second;
        state.pending.erase(it);

        auto response = parse_data<ActionResponse>(text, "action response");
        if (response.has_error()) {
            settle(*pending, outcome_error(std::move(response).error()));
        } else if (response->status != "ok" && response->status != "async") {
            settle(*pending, outcome_error(OneBotError{
                                 .kind = OneBotErrorKind::ACTION_FAILED,
                                 .detail = response->wording.value_or(response->message.value_or(response->status)),
                                 .retcode = response->retcode,
                             }));
        } else {
            settle(*pending, std::string(response->data.str));
        }
        return;
    }

    if (!frame.post_type) {
        log::debug("onebot: ignoring frame without echo or post_type");
        return;
    }
    auto event = decode_event(text, *frame.post_type);
    if (event.has_error()) {
        log::warn("onebot: ignoring undecodable {} event: {}", *frame.post_type, event.error().message());
        return;
    }
    on_event(std::move(*event));
}

void fail_all_pending(Session::State &state, std::string_view reason) {
    auto pending = std::move(state.pending);
    state.pending.clear();
    for (auto &[echo, waiter] : pending) {
        settle(*waiter, outcome_error(OneBotError{.kind = OneBotErrorKind::DISCONNECTED, .detail = std::string(reason)}));
    }
}

/// with_timeout() needs an Error-compatible error channel, so the websocket
/// outcome travels in the value instead.
Task<lighter::Outcome<net::WsMessage, net::WsError>> receive_outcome(net::WebSocket &socket) { co_return co_await socket.receive(); }

Task<void, OneBotError> run_impl(std::shared_ptr<Session> self, EventHandler on_event) {
    auto &state = *self->state;
    OneBotError ended;
    for (;;) {
        auto received = co_await lighter::with_timeout(receive_outcome(state.socket), state.options.idle_timeout);
        if (received.is_cancelled()) {
            state.open = false;
            state.socket.abort();
            fail_all_pending(state, "session cancelled");
            co_await lighter::cancel();
        }
        if (received.has_error()) {
            ended = {.kind = OneBotErrorKind::TIMEOUT,
                     .detail = "no frames for " + std::to_string(state.options.idle_timeout.count()) + "ms"};
            break;
        }
        auto message = std::move(*received);
        if (message.has_error()) {
            ended = transport(std::move(message).error());
            break;
        }
        if (message->binary) {
            log::debug("onebot: ignoring binary frame");
            continue;
        }
        dispatch(state, message->data, on_event);
    }

    state.open = false;
    state.socket.abort();
    fail_all_pending(state, ended.message());
    co_await lighter::fail(std::move(ended));
}

Task<std::string, OneBotError> call_impl(std::shared_ptr<Session> self, std::string action, std::string params_json) {
    auto &state = *self->state;
    if (!state.open) co_return outcome_error(OneBotError{.kind = OneBotErrorKind::DISCONNECTED, .detail = "session is closed"});

    std::string echo = "lilia-" + std::to_string(++state.next_echo);
    auto pending = std::make_shared<Pending>();
    state.pending.emplace(echo, pending);

    std::string request = R"({"action":)" + json_string(action) + R"(,"params":)" + (params_json.empty() ? "{}" : params_json) +
                          R"(,"echo":)" + json_string(echo) + "}";
    auto sent = co_await state.socket.send_text(std::move(request));
    if (sent.has_error()) {
        state.pending.erase(echo);
        co_return outcome_error(transport(std::move(sent).error()));
    }

    auto waited = co_await lighter::with_timeout(pending->done.wait(), state.options.action_timeout);
    if (waited.is_cancelled()) {
        state.pending.erase(echo);
        co_await lighter::cancel();
    }
    if (waited.has_error() || !pending->result) {
        state.pending.erase(echo);
        co_return outcome_error(OneBotError{.kind = OneBotErrorKind::TIMEOUT, .detail = action + " got no response"});
    }
    co_return std::move(*pending->result);
}

Task<LoginInfo, OneBotError> get_login_info_impl(std::shared_ptr<Session> self) {
    auto data = co_await call_impl(std::move(self), "get_login_info", "{}");
    if (data.has_error()) co_return outcome_error(std::move(data).error());
    co_return parse_data<LoginInfo>(*data, "get_login_info data");
}

template <typename Params>
Task<SentMessage, OneBotError> send_impl(std::shared_ptr<Session> self, std::string action, Params params) {
    auto encoded = glz::write_json(params);
    if (!encoded) co_return outcome_error(decode_error("cannot encode " + action + " params"));
    auto data = co_await call_impl(std::move(self), action, *std::move(encoded));
    if (data.has_error()) co_return outcome_error(std::move(data).error());
    co_return parse_data<SentMessage>(*data, action + " data");
}

} // namespace

Task<std::shared_ptr<Session>, OneBotError> Session::open(SessionOptions options) {
    auto connect = options.connect;
    if (!options.access_token.empty()) connect.headers.emplace_back("Authorization", "Bearer " + options.access_token);
    auto socket = co_await net::WebSocket::connect(std::move(connect));
    if (socket.has_error()) co_return outcome_error(transport(std::move(socket).error()));

    auto session = std::make_shared<Session>();
    session->state = std::make_unique<State>();
    session->state->socket = std::move(*socket);
    session->state->options = std::move(options);
    co_return std::move(session);
}

Task<void, OneBotError> Session::run(EventHandler on_event) { return run_impl(shared_from_this(), std::move(on_event)); }

Task<std::string, OneBotError> Session::call(std::string action, std::string params_json) {
    return call_impl(shared_from_this(), std::move(action), std::move(params_json));
}

Task<LoginInfo, OneBotError> Session::get_login_info() { return get_login_info_impl(shared_from_this()); }

Task<SentMessage, OneBotError> Session::send_group_msg(i64 group_id, Message message) {
    return send_impl(shared_from_this(), "send_group_msg", SendGroupParams{.group_id = group_id, .message = std::move(message)});
}

Task<SentMessage, OneBotError> Session::send_private_msg(i64 user_id, Message message) {
    return send_impl(shared_from_this(), "send_private_msg", SendPrivateParams{.user_id = user_id, .message = std::move(message)});
}

namespace {

struct HistoryPage {
    std::vector<MessageEvent> messages;
};

} // namespace

// With `message_seq` (a message id) NapCat pages forward from that message
// unless `reverseOrder` is set, which pages back up to it.
Task<std::vector<MessageEvent>> group_history(std::shared_ptr<Session> session, i64 group, usize count) {
    constexpr usize k_page = 500;
    std::vector<MessageEvent> collected;
    std::set<i64> seen;
    std::optional<i64> anchor;
    while (collected.size() < count) {
        auto params = std::format(R"({{"group_id":{},"count":{})", group, std::min(k_page, count - collected.size() + 1));
        if (anchor) params += std::format(R"(,"message_seq":{},"reverseOrder":true)", *anchor);
        params += "}";
        auto data = co_await session->call("get_group_msg_history", std::move(params));
        if (data.has_error()) {
            log::warn("group {}: cannot read history: {}", group, data.error().message());
            break;
        }
        HistoryPage page;
        if (auto ec = glz::read<k_read_opts>(page, *data)) {
            log::warn("group {}: undecodable history: {}", group, glz::format_error(ec, *data));
            break;
        }
        std::vector<MessageEvent> older;
        for (auto &message : page.messages) {
            if (seen.insert(message.message_id).second) older.push_back(std::move(message));
        }
        if (older.empty()) break;
        std::ranges::sort(older, {}, &MessageEvent::time);
        anchor = older.front().message_id;
        collected.insert(collected.begin(), std::make_move_iterator(older.begin()), std::make_move_iterator(older.end()));
    }
    std::ranges::stable_sort(collected, {}, &MessageEvent::time);
    if (collected.size() > count) collected.erase(collected.begin(), collected.end() - static_cast<std::ptrdiff_t>(count));
    co_return collected;
}

void Session::close() noexcept {
    if (!state || !state->open) return;
    state->open = false;
    state->socket.abort();
}

bool Session::is_open() const noexcept { return state && state->open; }

} // namespace lilia::onebot
