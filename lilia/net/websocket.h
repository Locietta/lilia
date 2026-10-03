#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <lighter/types.hpp>
#include <lighter/async/runtime/task.h>
#include <lighter/async/vocab/error.h>
#include <lighter/async/vocab/outcome.h>

namespace lilia::net {

using namespace lighter::types;

// Minimal RFC 6455 client for plain ws:// endpoints on a trusted network
// (NapCat's OneBot server on loopback). There is no TLS and no extension
// negotiation; host names must be numeric because lighter::Tcp does not
// resolve names.

enum struct WsErrorKind {
    IO,        ///< socket failure; see `io`
    HANDSHAKE, ///< the HTTP upgrade was rejected or malformed
    PROTOCOL,  ///< the peer violated the framing rules
    TOO_LARGE, ///< a message exceeded the configured limit
    CLOSED,    ///< the connection closed; see `close_code`
};

struct WsError {
    WsErrorKind kind = WsErrorKind::PROTOCOL;
    std::string detail;
    lighter::Error io;
    /// Peer status code for CLOSED; 1005 when the close frame carried none and
    /// 1006 when the TCP stream ended without a close frame.
    u16 close_code = 0;

    std::string message() const;

    static WsError io_error(lighter::Error error, std::string detail) {
        return {.kind = WsErrorKind::IO, .detail = std::move(detail), .io = error};
    }
    static WsError handshake(std::string detail) { return {.kind = WsErrorKind::HANDSHAKE, .detail = std::move(detail)}; }
    static WsError protocol(std::string detail) { return {.kind = WsErrorKind::PROTOCOL, .detail = std::move(detail)}; }
    static WsError closed(u16 code, std::string reason) {
        return {.kind = WsErrorKind::CLOSED, .detail = std::move(reason), .close_code = code};
    }
};

using Header = std::pair<std::string, std::string>;

namespace ws {

enum struct Opcode : u8 {
    CONTINUATION = 0x0,
    TEXT = 0x1,
    BINARY = 0x2,
    CLOSE = 0x8,
    PING = 0x9,
    PONG = 0xA,
};

struct Frame {
    bool fin = true;
    Opcode opcode = Opcode::TEXT;
    std::string payload;
};

inline constexpr usize k_default_max_message_bytes = 16 * 1024 * 1024;

/// Encodes one client frame. Client frames are always masked (RFC 6455 5.3).
std::string encode_client_frame(Opcode opcode, std::string_view payload, std::array<u8, 4> mask, bool fin = true);

/// Decodes the frame at the front of `buffer` and erases its bytes. Returns
/// nullopt, leaving `buffer` untouched, while the frame is still incomplete.
lighter::Outcome<std::optional<Frame>, WsError> decode_frame(std::string &buffer, usize max_payload_bytes);

/// Sec-WebSocket-Accept value the server must answer for `client_key`.
std::string accept_key(std::string_view client_key);

std::string make_handshake_request(std::string_view host, i32 port, std::string_view path, std::string_view client_key,
                                   const std::vector<Header> &headers);

/// Validates a response head (status line and headers, without the blank
/// line) against the accept key derived from the request.
lighter::Outcome<void, WsError> check_handshake_response(std::string_view head, std::string_view expected_accept);

} // namespace ws

struct WsMessage {
    bool binary = false;
    std::string data;
};

struct WsConnectOptions {
    std::string host = "127.0.0.1";
    i32 port = 80;
    std::string path = "/";
    std::vector<Header> headers;
    usize max_message_bytes = ws::k_default_max_message_bytes;
};

struct WebSocket {
    WebSocket() noexcept;
    WebSocket(WebSocket &&) noexcept;
    WebSocket &operator=(WebSocket &&) noexcept;
    ~WebSocket();

    static lighter::Task<WebSocket, WsError> connect(WsConnectOptions options);

    /// Next complete data message. Pings are answered and pongs skipped
    /// internally. Only one receive() may be pending at a time.
    lighter::Task<WsMessage, WsError> receive();

    /// Sends are serialized internally, so any task may call these while
    /// another task is blocked in receive().
    lighter::Task<void, WsError> send_text(std::string text);
    lighter::Task<void, WsError> send_ping(std::string payload = {});

    /// Sends a close frame. The connection is finished once receive() reports
    /// the peer's close (or the stream ends).
    lighter::Task<void, WsError> close(u16 code = 1000, std::string reason = {});

    /// Drops the TCP connection without a closing handshake and wakes a
    /// pending receive().
    void abort() noexcept;

    bool is_open() const noexcept;

    struct State;
    std::unique_ptr<State> state;
};

} // namespace lilia::net
