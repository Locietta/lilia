#include <array>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include <lighter/async/io/loop.h>
#include <lighter/async/io/stream.h>
#include <lighter/async/runtime/task.h>

#include <lilia/net/codec.h>
#include <lilia/net/websocket.h>

namespace {

using namespace lilia::net;
using namespace lighter::types;

void require(bool condition, std::string message) {
    if (!condition) {
        throw std::runtime_error(std::move(message));
    }
}

/// Server frames are never masked.
std::string server_frame(ws::Opcode opcode, std::string_view payload, bool fin = true) {
    std::string out;
    out.push_back(static_cast<char>((fin ? 0x80 : 0) | static_cast<u8>(opcode)));
    if (payload.size() < 126) {
        out.push_back(static_cast<char>(payload.size()));
    } else if (payload.size() <= 0xffff) {
        out.push_back(static_cast<char>(126));
        out.push_back(static_cast<char>(payload.size() >> 8));
        out.push_back(static_cast<char>(payload.size() & 0xff));
    } else {
        out.push_back(static_cast<char>(127));
        for (i32 shift = 56; shift >= 0; shift -= 8) out.push_back(static_cast<char>(static_cast<u64>(payload.size()) >> shift & 0xff));
    }
    out += payload;
    return out;
}

void test_codecs() {
    require(base64_encode("") == "", "base64 of empty input");
    require(base64_encode("f") == "Zg==", "base64 one byte");
    require(base64_encode("fo") == "Zm8=", "base64 two bytes");
    require(base64_encode("foo") == "Zm9v", "base64 three bytes");
    require(base64_encode("foobar") == "Zm9vYmFy", "base64 six bytes");
    require(base64_decode("Zm9vYmFy") == "foobar" && base64_decode("Zg==") == "f" && base64_decode("Zm8") == "fo",
            "base64 decodes, padding optional");
    require(base64_decode("Zm9v\nYmFy") == "foobar" && !base64_decode("Zm9v!"), "whitespace skipped, junk refused");
    require(base64_decode(base64_encode("群公告：周六开黑")) == "群公告：周六开黑", "round trip");

    const auto digest = sha1("abc");
    constexpr std::array<u8, 20> expected{0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a, 0xba, 0x3e,
                                          0x25, 0x71, 0x78, 0x50, 0xc2, 0x6c, 0x9c, 0xd0, 0xd8, 0x9d};
    require(digest == expected, "sha1(\"abc\") test vector");
    // Fed in pieces of every size, across block boundaries, the digest is the same.
    std::string long_input;
    for (int i = 0; i < 1000; ++i) long_input += static_cast<char>('a' + i % 26);
    const auto whole = sha1(long_input);
    for (const usize piece : {1, 7, 63, 64, 65, 333}) {
        Sha1 streamed;
        for (usize at = 0; at < long_input.size(); at += piece) streamed.update(std::string_view(long_input).substr(at, piece));
        require(streamed.finish() == whole, "streamed sha1 in pieces of " + std::to_string(piece));
    }

    // RFC 6455 section 1.3 example.
    require(ws::accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", "RFC 6455 accept key example");
}

void test_frame_round_trips() {
    for (usize size : {usize{0}, usize{5}, usize{125}, usize{126}, usize{65535}, usize{65536}}) {
        std::string payload(size, 'x');
        for (usize i = 0; i < size; ++i) payload[i] = static_cast<char>('a' + i % 26);

        auto encoded = ws::encode_client_frame(ws::Opcode::TEXT, payload, {0x12, 0x34, 0x56, 0x78});
        require((static_cast<u8>(encoded[1]) & 0x80) != 0, "client frames must be masked");

        // Feed byte by byte up to the last one: every prefix is incomplete.
        std::string buffer;
        for (usize i = 0; i + 1 < encoded.size(); i += std::max<usize>(1, encoded.size() / 7)) {
            buffer.assign(encoded, 0, i);
            auto partial = ws::decode_frame(buffer, ws::k_default_max_message_bytes);
            require(partial.has_value() && !*partial, "an incomplete frame must not decode");
            require(buffer.size() == i, "an incomplete frame must not consume input");
        }

        buffer = encoded + "trailing";
        auto decoded = ws::decode_frame(buffer, ws::k_default_max_message_bytes);
        require(decoded.has_value() && *decoded, "a complete frame must decode");
        require((*decoded)->payload == payload, "masked payload must round-trip at size " + std::to_string(size));
        require((*decoded)->fin && (*decoded)->opcode == ws::Opcode::TEXT, "frame header must round-trip");
        require(buffer == "trailing", "decoding must consume exactly one frame");
    }
}

void test_frame_rejections() {
    std::string reserved("\xC1\x00", 2);
    require(ws::decode_frame(reserved, 1024).has_error(), "reserved bits must be rejected");

    std::string unknown("\x83\x00", 2);
    require(ws::decode_frame(unknown, 1024).has_error(), "unknown opcodes must be rejected");

    std::string fragmented_ping = server_frame(ws::Opcode::PING, "x", false);
    require(ws::decode_frame(fragmented_ping, 1024).has_error(), "fragmented control frames must be rejected");

    std::string long_ping = server_frame(ws::Opcode::PING, std::string(126, 'p'));
    require(ws::decode_frame(long_ping, 1024).has_error(), "control frames over 125 bytes must be rejected");

    std::string large = server_frame(ws::Opcode::TEXT, std::string(2000, 'l'));
    auto too_large = ws::decode_frame(large, 1024);
    require(too_large.has_error() && too_large.error().kind == WsErrorKind::TOO_LARGE, "oversized frames must be rejected");
}

void test_handshake_check() {
    const auto accept = ws::accept_key("dGhlIHNhbXBsZSBub25jZQ==");
    const std::string good = "HTTP/1.1 101 Switching Protocols\r\nupgrade: WebSocket\r\nConnection: keep-alive, Upgrade\r\n"
                             "Sec-WebSocket-Accept: " +
                             accept;
    require(ws::check_handshake_response(good, accept).has_value(), "a valid upgrade response must pass");

    const std::string rejected = "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0";
    require(ws::check_handshake_response(rejected, accept).has_error(), "a non-101 status must fail");

    const std::string wrong_key = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                                  "Sec-WebSocket-Accept: AAAA";
    require(ws::check_handshake_response(wrong_key, accept).has_error(), "a wrong accept key must fail");

    const auto request = ws::make_handshake_request("127.0.0.1", 3001, "/", "key==", {{"Authorization", "Bearer t"}});
    require(request.starts_with("GET / HTTP/1.1\r\nHost: 127.0.0.1:3001\r\n"), "request line and host header");
    require(request.find("\r\nAuthorization: Bearer t\r\n") != std::string::npos, "extra headers must be sent");
    require(request.ends_with("\r\n\r\n"), "request head must end with a blank line");
}

/// Reads one masked client frame from the connection.
lighter::Task<ws::Frame> read_client_frame(lighter::Tcp &connection, std::string &buffer) {
    for (;;) {
        auto decoded = ws::decode_frame(buffer, ws::k_default_max_message_bytes);
        require(decoded.has_value(), "client sent an invalid frame");
        if (*decoded) co_return std::move(**decoded);
        auto chunk = co_await connection.read();
        require(static_cast<bool>(chunk) && !chunk->empty(), "client connection ended early");
        buffer += *chunk;
    }
}

lighter::Task<> write_all(lighter::Tcp &connection, std::string_view bytes) {
    auto written = co_await connection.write(std::span(bytes.data(), bytes.size()));
    require(static_cast<bool>(written), "server write failed");
}

lighter::Task<> fake_server(lighter::Tcp::Acceptor listener) {
    auto accepted = co_await listener.accept();
    require(static_cast<bool>(accepted), "failed to accept the websocket test connection");
    listener = {};
    auto connection = std::move(*accepted);

    std::string buffer;
    while (buffer.find("\r\n\r\n") == std::string::npos) {
        auto chunk = co_await connection.read();
        require(static_cast<bool>(chunk) && !chunk->empty(), "handshake request ended early");
        buffer += *chunk;
    }
    const auto head_end = buffer.find("\r\n\r\n");
    const std::string head = buffer.substr(0, head_end + 2); // keep the last header's CRLF
    buffer.erase(0, head_end + 4);
    require(head.find("\r\nAuthorization: Bearer secret\r\n") != std::string::npos, "client must send the bearer token");
    const auto key_pos = head.find("Sec-WebSocket-Key: ");
    require(key_pos != std::string::npos, "client must send a key");
    const auto key = head.substr(key_pos + 19, head.find("\r\n", key_pos) - key_pos - 19);

    // Response head and the first frames arrive in one write, so the client
    // must keep bytes that follow the head.
    std::string response =
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " + ws::accept_key(key) +
        "\r\n\r\n";
    response += server_frame(ws::Opcode::TEXT, "hello");
    response += server_frame(ws::Opcode::PING, "are you there");
    response += server_frame(ws::Opcode::TEXT, "frag", false);
    response += server_frame(ws::Opcode::CONTINUATION, "mented");
    co_await write_all(connection, response);

    auto pong = co_await read_client_frame(connection, buffer);
    require(pong.opcode == ws::Opcode::PONG && pong.payload == "are you there", "client must answer a ping with its payload");

    auto echo = co_await read_client_frame(connection, buffer);
    require(echo.opcode == ws::Opcode::TEXT && echo.payload == "hello|fragmented", "client text must arrive unmasked by the decoder");

    std::string close_payload = "\x03\xE8";
    close_payload += "bye";
    co_await write_all(connection, server_frame(ws::Opcode::CLOSE, close_payload));

    auto close_echo = co_await read_client_frame(connection, buffer);
    require(close_echo.opcode == ws::Opcode::CLOSE && close_echo.payload == "\x03\xE8", "client must echo the close code");
}

lighter::Task<> client(i32 port) {
    auto connected = co_await WebSocket::connect({
        .host = "127.0.0.1",
        .port = port,
        .path = "/",
        .headers = {{"Authorization", "Bearer secret"}},
    });
    require(connected.has_value(), connected ? std::string() : connected.error().message());
    auto socket = std::move(*connected);

    auto first = co_await socket.receive();
    require(first.has_value() && first->data == "hello", "first message");
    auto second = co_await socket.receive();
    require(second.has_value() && second->data == "fragmented", "fragmented message must be reassembled");

    auto sent = co_await socket.send_text(first->data + "|" + second->data);
    require(sent.has_value(), "send_text must succeed");

    auto closed = co_await socket.receive();
    require(closed.has_error() && closed.error().kind == WsErrorKind::CLOSED && closed.error().close_code == 1000 &&
                closed.error().detail == "bye",
            "peer close must surface as CLOSED with its code and reason");
    require(!socket.is_open(), "socket must be closed after the closing handshake");
}

void test_loopback_exchange() {
    lighter::EventLoop loop;
    auto listener = lighter::Tcp::listen("127.0.0.1", 0, {}, loop);
    require(static_cast<bool>(listener), "failed to create the websocket test listener");
    auto port = lighter::Tcp::local_port(*listener);
    require(static_cast<bool>(port), "failed to query the websocket test listener port");

    auto server_task = fake_server(std::move(*listener));
    auto client_task = client(*port);
    loop.schedule(server_task);
    loop.schedule(client_task);
    loop.run();

    std::ignore = server_task.value();
    std::ignore = client_task.value();
}

} // namespace

int main() {
    try {
        test_codecs();
        test_frame_round_trips();
        test_frame_rejections();
        test_handshake_check();
        test_loopback_exchange();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
