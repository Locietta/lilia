#include "websocket.h"

#include <algorithm>
#include <cctype>
#include <random>
#include <span>
#include <tuple>

#include <lighter/async/io/stream.h>
#include <lighter/async/runtime/sync.h>

#include <lilia/net/codec.h>
#include <lilia/text.h>

namespace lilia::net {

using lighter::outcome_error;
using lighter::Task;

namespace {

constexpr std::string_view k_accept_guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
constexpr usize k_max_handshake_bytes = 16 * 1024;
constexpr usize k_read_chunk_bytes = 16 * 1024;

bool is_control(ws::Opcode opcode) { return (static_cast<u8>(opcode) & 0x8) != 0; }

bool known_opcode(u8 value) {
    switch (value) {
        case 0x0:
        case 0x1:
        case 0x2:
        case 0x8:
        case 0x9:
        case 0xA: return true;
        default: return false;
    }
}

bool equals_ignore_case(std::string_view a, std::string_view b) {
    return std::ranges::equal(
        a, b, [](char x, char y) { return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y)); });
}

/// True when the comma-separated header value lists `token` (case-insensitive).
bool header_has_token(std::string_view value, std::string_view token) {
    while (!value.empty()) {
        const auto comma = value.find(',');
        if (equals_ignore_case(trim(value.substr(0, comma)), token)) return true;
        if (comma == std::string_view::npos) break;
        value.remove_prefix(comma + 1);
    }
    return false;
}

} // namespace

std::string WsError::message() const {
    std::string out;
    switch (kind) {
        case WsErrorKind::IO: out = "websocket io error: "; break;
        case WsErrorKind::HANDSHAKE: out = "websocket handshake failed: "; break;
        case WsErrorKind::PROTOCOL: out = "websocket protocol error: "; break;
        case WsErrorKind::TOO_LARGE: out = "websocket message too large: "; break;
        case WsErrorKind::CLOSED: out = "websocket closed (" + std::to_string(close_code) + "): "; break;
    }
    out += detail;
    if (io) {
        out += " (";
        out += io.message();
        out += ")";
    }
    return out;
}

namespace ws {

std::string encode_client_frame(Opcode opcode, std::string_view payload, std::array<u8, 4> mask, bool fin) {
    std::string out;
    out.reserve(payload.size() + 14);
    out.push_back(static_cast<char>((fin ? 0x80 : 0x00) | static_cast<u8>(opcode)));
    const u64 size = payload.size();
    if (size < 126) {
        out.push_back(static_cast<char>(0x80 | size));
    } else if (size <= 0xffff) {
        out.push_back(static_cast<char>(0x80 | 126));
        out.push_back(static_cast<char>(size >> 8 & 0xff));
        out.push_back(static_cast<char>(size & 0xff));
    } else {
        out.push_back(static_cast<char>(0x80 | 127));
        for (i32 shift = 56; shift >= 0; shift -= 8) out.push_back(static_cast<char>(size >> shift & 0xff));
    }
    for (u8 byte : mask) out.push_back(static_cast<char>(byte));
    for (usize i = 0; i < payload.size(); ++i) out.push_back(static_cast<char>(static_cast<u8>(payload[i]) ^ mask[i % 4]));
    return out;
}

lighter::Outcome<std::optional<Frame>, WsError> decode_frame(std::string &buffer, usize max_payload_bytes) {
    if (buffer.size() < 2) return std::optional<Frame>{};
    const auto byte = [&buffer](usize index) { return static_cast<u8>(buffer[index]); };

    const u8 first = byte(0);
    const u8 second = byte(1);
    if ((first & 0x70) != 0) return outcome_error(WsError::protocol("reserved bits set without a negotiated extension"));
    const u8 opcode_value = first & 0x0f;
    if (!known_opcode(opcode_value)) return outcome_error(WsError::protocol("unknown opcode " + std::to_string(opcode_value)));

    Frame frame{.fin = (first & 0x80) != 0, .opcode = static_cast<Opcode>(opcode_value)};
    const bool masked = (second & 0x80) != 0;
    u64 size = second & 0x7f;
    usize offset = 2;
    if (size == 126) {
        if (buffer.size() < 4) return std::optional<Frame>{};
        size = static_cast<u64>(byte(2)) << 8 | byte(3);
        offset = 4;
    } else if (size == 127) {
        if (buffer.size() < 10) return std::optional<Frame>{};
        size = 0;
        for (usize i = 2; i < 10; ++i) size = size << 8 | byte(i);
        if (size >> 63) return outcome_error(WsError::protocol("payload length has the most significant bit set"));
        offset = 10;
    }

    if (is_control(frame.opcode)) {
        if (!frame.fin) return outcome_error(WsError::protocol("fragmented control frame"));
        if (size > 125) return outcome_error(WsError::protocol("control frame payload longer than 125 bytes"));
    }
    if (size > max_payload_bytes) {
        return outcome_error(WsError{.kind = WsErrorKind::TOO_LARGE, .detail = std::to_string(size) + " byte frame"});
    }

    std::array<u8, 4> mask{};
    if (masked) {
        if (buffer.size() < offset + 4) return std::optional<Frame>{};
        for (usize i = 0; i < 4; ++i) mask[i] = byte(offset + i);
        offset += 4;
    }
    if (buffer.size() - offset < size) return std::optional<Frame>{};

    frame.payload.assign(buffer, offset, static_cast<usize>(size));
    if (masked) {
        for (usize i = 0; i < frame.payload.size(); ++i)
            frame.payload[i] = static_cast<char>(static_cast<u8>(frame.payload[i]) ^ mask[i % 4]);
    }
    buffer.erase(0, offset + static_cast<usize>(size));
    return std::optional<Frame>(std::move(frame));
}

std::string accept_key(std::string_view client_key) {
    std::string input(client_key);
    input += k_accept_guid;
    const auto digest = sha1(input);
    return base64_encode(std::string_view(reinterpret_cast<const char *>(digest.data()), digest.size()));
}

std::string make_handshake_request(std::string_view host, i32 port, std::string_view path, std::string_view client_key,
                                   const std::vector<Header> &headers) {
    std::string request = "GET ";
    request += path.empty() ? "/" : path;
    request += " HTTP/1.1\r\nHost: ";
    // IPv6 literals need brackets in the Host header.
    if (host.find(':') != std::string_view::npos) {
        request += '[';
        request += host;
        request += ']';
    } else {
        request += host;
    }
    request += ':' + std::to_string(port);
    request += "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: ";
    request += client_key;
    request += "\r\n";
    for (const auto &[name, value] : headers) {
        request += name;
        request += ": ";
        request += value;
        request += "\r\n";
    }
    request += "\r\n";
    return request;
}

lighter::Outcome<void, WsError> check_handshake_response(std::string_view head, std::string_view expected_accept) {
    const auto line_end = head.find("\r\n");
    const auto status_line = head.substr(0, line_end);
    // "HTTP/1.1 101 Switching Protocols"
    const auto first_space = status_line.find(' ');
    if (!status_line.starts_with("HTTP/1.1 ") || first_space == std::string_view::npos || status_line.substr(first_space + 1, 3) != "101") {
        return outcome_error(WsError::handshake("unexpected status: " + std::string(status_line)));
    }

    bool upgrade = false;
    bool connection = false;
    bool accepted = false;
    auto rest = line_end == std::string_view::npos ? std::string_view{} : head.substr(line_end + 2);
    while (!rest.empty()) {
        const auto end = rest.find("\r\n");
        const auto line = rest.substr(0, end);
        rest = end == std::string_view::npos ? std::string_view{} : rest.substr(end + 2);
        const auto colon = line.find(':');
        if (colon == std::string_view::npos) continue;
        const auto name = trim(line.substr(0, colon));
        const auto value = trim(line.substr(colon + 1));
        if (equals_ignore_case(name, "Upgrade")) upgrade = equals_ignore_case(value, "websocket");
        if (equals_ignore_case(name, "Connection")) connection = header_has_token(value, "Upgrade");
        if (equals_ignore_case(name, "Sec-WebSocket-Accept")) accepted = value == expected_accept;
    }
    if (!upgrade || !connection) return outcome_error(WsError::handshake("response is not a websocket upgrade"));
    if (!accepted) return outcome_error(WsError::handshake("Sec-WebSocket-Accept does not match the request key"));
    return {};
}

} // namespace ws

struct WebSocket::State {
    lighter::Tcp tcp;
    std::string inbox;
    lighter::Mutex write_lock;
    std::mt19937 rng{std::random_device{}()};
    usize max_message_bytes = ws::k_default_max_message_bytes;
    bool open = true;
    bool close_sent = false;

    std::array<u8, 4> next_mask() {
        const u32 bits = static_cast<u32>(rng());
        return {static_cast<u8>(bits), static_cast<u8>(bits >> 8), static_cast<u8>(bits >> 16), static_cast<u8>(bits >> 24)};
    }
};

namespace {

/// Releases the write lock even when the writing task is torn down mid-write.
struct WriteLockGuard {
    explicit WriteLockGuard(lighter::Mutex &mutex) : mutex(&mutex) {}

    WriteLockGuard(const WriteLockGuard &) = delete;
    WriteLockGuard &operator=(const WriteLockGuard &) = delete;

    ~WriteLockGuard() { mutex->unlock(); }

    lighter::Mutex *mutex;
};

Task<void, WsError> write_frame(WebSocket::State *state, ws::Opcode opcode, std::string payload) {
    if (!state->open) co_await lighter::fail(WsError::closed(1006, "connection is not open"));
    co_await state->write_lock.lock();
    WriteLockGuard guard(state->write_lock);
    const auto frame = ws::encode_client_frame(opcode, payload, state->next_mask());
    auto written = co_await state->tcp.write(std::span<const char>(frame.data(), frame.size()));
    if (written.has_error()) {
        state->open = false;
        co_await lighter::fail(WsError::io_error(written.error(), "write failed"));
    }
}

std::string random_key(std::mt19937 &rng) {
    std::string nonce(16, '\0');
    for (auto &byte : nonce) byte = static_cast<char>(rng() & 0xff);
    return base64_encode(nonce);
}

void abort_state(WebSocket::State *state) noexcept {
    if (!state || !state->open) return;
    state->open = false;
    state->tcp.stop();
}

} // namespace

WebSocket::WebSocket() noexcept = default;
WebSocket::WebSocket(WebSocket &&) noexcept = default;
WebSocket &WebSocket::operator=(WebSocket &&) noexcept = default;
WebSocket::~WebSocket() = default;

Task<WebSocket, WsError> WebSocket::connect(WsConnectOptions options) {
    auto state = std::make_unique<State>();
    state->max_message_bytes = options.max_message_bytes;

    auto tcp = co_await lighter::Tcp::connect(options.host, options.port);
    if (tcp.has_error())
        co_return outcome_error(WsError::io_error(tcp.error(), "connect to " + options.host + ":" + std::to_string(options.port)));
    state->tcp = std::move(*tcp);

    const auto key = random_key(state->rng);
    const auto request = ws::make_handshake_request(options.host, options.port, options.path, key, options.headers);
    auto sent = co_await state->tcp.write(std::span<const char>(request.data(), request.size()));
    if (sent.has_error()) co_return outcome_error(WsError::io_error(sent.error(), "send handshake"));

    std::array<char, k_read_chunk_bytes> chunk{};
    usize head_end = std::string::npos;
    while ((head_end = state->inbox.find("\r\n\r\n")) == std::string::npos) {
        if (state->inbox.size() > k_max_handshake_bytes) co_return outcome_error(WsError::handshake("response head too large"));
        auto read = co_await state->tcp.read_some(chunk);
        if (read.has_error()) co_return outcome_error(WsError::io_error(read.error(), "read handshake"));
        if (*read == 0) co_return outcome_error(WsError::handshake("connection closed during handshake"));
        state->inbox.append(chunk.data(), *read);
    }

    auto checked = ws::check_handshake_response(std::string_view(state->inbox).substr(0, head_end), ws::accept_key(key));
    if (checked.has_error()) co_return outcome_error(std::move(checked).error());
    // Frames may already follow the response head in the same read.
    state->inbox.erase(0, head_end + 4);

    WebSocket socket;
    socket.state = std::move(state);
    co_return std::move(socket);
}

namespace {

Task<WsMessage, WsError> receive_impl(WebSocket::State *s) {
    std::string message;
    std::optional<ws::Opcode> fragmented;
    std::array<char, k_read_chunk_bytes> chunk{};

    for (;;) {
        auto decoded = ws::decode_frame(s->inbox, s->max_message_bytes);
        if (decoded.has_error()) {
            abort_state(s);
            co_return outcome_error(std::move(decoded).error());
        }
        if (!*decoded) {
            if (!s->open) co_return outcome_error(WsError::closed(1006, "connection aborted"));
            auto read = co_await s->tcp.read_some(chunk);
            if (read.has_error()) {
                s->open = false;
                co_return outcome_error(WsError::io_error(read.error(), "read failed"));
            }
            if (*read == 0) {
                s->open = false;
                co_return outcome_error(WsError::closed(1006, "stream ended without a close frame"));
            }
            s->inbox.append(chunk.data(), *read);
            continue;
        }

        auto frame = std::move(**decoded);
        switch (frame.opcode) {
            case ws::Opcode::PING: {
                auto pong = co_await write_frame(s, ws::Opcode::PONG, std::move(frame.payload));
                if (pong.has_error()) co_return outcome_error(std::move(pong).error());
                continue;
            }
            case ws::Opcode::PONG: continue;
            case ws::Opcode::CLOSE: {
                if (frame.payload.size() == 1) {
                    abort_state(s);
                    co_return outcome_error(WsError::protocol("close frame with a one-byte payload"));
                }
                u16 code = 1005;
                std::string reason;
                if (frame.payload.size() >= 2) {
                    code = static_cast<u16>(static_cast<u8>(frame.payload[0]) << 8 | static_cast<u8>(frame.payload[1]));
                    reason = frame.payload.substr(2);
                }
                if (!s->close_sent) {
                    s->close_sent = true;
                    std::ignore = co_await write_frame(s, ws::Opcode::CLOSE, frame.payload.substr(0, 2));
                }
                // The server closes TCP after the handshake; nothing useful
                // can arrive anymore.
                abort_state(s);
                co_return outcome_error(WsError::closed(code, std::move(reason)));
            }
            case ws::Opcode::TEXT:
            case ws::Opcode::BINARY:
                if (fragmented) {
                    abort_state(s);
                    co_return outcome_error(WsError::protocol("data frame inside a fragmented message"));
                }
                if (frame.fin) co_return WsMessage{.binary = frame.opcode == ws::Opcode::BINARY, .data = std::move(frame.payload)};
                fragmented = frame.opcode;
                message = std::move(frame.payload);
                continue;
            case ws::Opcode::CONTINUATION:
                if (!fragmented) {
                    abort_state(s);
                    co_return outcome_error(WsError::protocol("continuation frame without a message"));
                }
                if (message.size() + frame.payload.size() > s->max_message_bytes) {
                    abort_state(s);
                    co_return outcome_error(WsError{.kind = WsErrorKind::TOO_LARGE, .detail = "fragmented message"});
                }
                message += frame.payload;
                if (frame.fin) co_return WsMessage{.binary = *fragmented == ws::Opcode::BINARY, .data = std::move(message)};
                continue;
        }
    }
}

} // namespace

Task<WsMessage, WsError> WebSocket::receive() { return receive_impl(state.get()); }

Task<void, WsError> WebSocket::send_text(std::string text) { return write_frame(state.get(), ws::Opcode::TEXT, std::move(text)); }

void WebSocket::abort() noexcept { abort_state(state.get()); }

bool WebSocket::is_open() const noexcept { return state && state->open; }

} // namespace lilia::net
