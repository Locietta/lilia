#pragma once

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <lighter/async/io/stream.h>
#include <lighter/async/runtime/task.h>

#include <lilia/net/websocket.h>

namespace lilia::test {

using namespace lighter::types;

inline void require(bool condition, std::string message) {
    if (!condition) throw std::runtime_error(std::move(message));
}

/// Server frames are never masked.
inline std::string server_frame(net::ws::Opcode opcode, std::string_view payload, bool fin = true) {
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

/// Server side of one accepted websocket connection, for tests.
struct FakeWsPeer {
    lighter::Tcp connection;
    std::string buffer;
    std::string request_head;

    static lighter::Task<FakeWsPeer> accept(lighter::Tcp::Acceptor &listener) {
        auto accepted = co_await listener.accept();
        require(static_cast<bool>(accepted), "failed to accept the test connection");
        FakeWsPeer peer{.connection = std::move(*accepted)};

        while (peer.buffer.find("\r\n\r\n") == std::string::npos) {
            auto chunk = co_await peer.connection.read();
            require(static_cast<bool>(chunk) && !chunk->empty(), "handshake request ended early");
            peer.buffer += *chunk;
        }
        const auto head_end = peer.buffer.find("\r\n\r\n");
        peer.request_head = peer.buffer.substr(0, head_end + 2);
        peer.buffer.erase(0, head_end + 4);

        const auto key_pos = peer.request_head.find("Sec-WebSocket-Key: ");
        require(key_pos != std::string::npos, "client must send a key");
        const auto key = peer.request_head.substr(key_pos + 19, peer.request_head.find("\r\n", key_pos) - key_pos - 19);
        co_await peer.write("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " +
                            net::ws::accept_key(key) + "\r\n\r\n");
        co_return std::move(peer);
    }

    bool has_header(std::string_view line) const { return request_head.find("\r\n" + std::string(line) + "\r\n") != std::string::npos; }

    lighter::Task<> write(std::string bytes) {
        auto written = co_await connection.write(std::span<const char>(bytes.data(), bytes.size()));
        require(static_cast<bool>(written), "fake peer write failed");
    }

    lighter::Task<> send_text(std::string_view text) { co_await write(server_frame(net::ws::Opcode::TEXT, text)); }

    lighter::Task<> send_close(u16 code = 1000) {
        std::string payload;
        payload.push_back(static_cast<char>(code >> 8));
        payload.push_back(static_cast<char>(code & 0xff));
        co_await write(server_frame(net::ws::Opcode::CLOSE, payload));
    }

    /// Next client data frame; control frames other than close are skipped.
    lighter::Task<net::ws::Frame> read_frame() {
        for (;;) {
            auto decoded = net::ws::decode_frame(buffer, net::ws::k_default_max_message_bytes);
            require(decoded.has_value(), "client sent an invalid frame");
            if (*decoded) {
                if ((*decoded)->opcode == net::ws::Opcode::PONG) continue;
                co_return std::move(**decoded);
            }
            auto chunk = co_await connection.read();
            require(static_cast<bool>(chunk) && !chunk->empty(), "client connection ended early");
            buffer += *chunk;
        }
    }

    lighter::Task<std::string> read_text() {
        auto frame = co_await read_frame();
        require(frame.opcode == net::ws::Opcode::TEXT, "expected a text frame from the client");
        co_return std::move(frame.payload);
    }
};

} // namespace lilia::test
