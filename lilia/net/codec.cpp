#include "codec.h"

#include <algorithm>
#include <bit>

namespace lilia::net {

std::string base64_encode(std::string_view bytes) {
    static constexpr std::string_view k_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    usize i = 0;
    for (; i + 3 <= bytes.size(); i += 3) {
        const u32 group = static_cast<u32>(static_cast<u8>(bytes[i])) << 16 | static_cast<u32>(static_cast<u8>(bytes[i + 1])) << 8 |
                          static_cast<u32>(static_cast<u8>(bytes[i + 2]));
        out.push_back(k_alphabet[group >> 18 & 0x3f]);
        out.push_back(k_alphabet[group >> 12 & 0x3f]);
        out.push_back(k_alphabet[group >> 6 & 0x3f]);
        out.push_back(k_alphabet[group & 0x3f]);
    }
    const usize rest = bytes.size() - i;
    if (rest == 1) {
        const u32 group = static_cast<u32>(static_cast<u8>(bytes[i])) << 16;
        out.push_back(k_alphabet[group >> 18 & 0x3f]);
        out.push_back(k_alphabet[group >> 12 & 0x3f]);
        out.append("==");
    } else if (rest == 2) {
        const u32 group = static_cast<u32>(static_cast<u8>(bytes[i])) << 16 | static_cast<u32>(static_cast<u8>(bytes[i + 1])) << 8;
        out.push_back(k_alphabet[group >> 18 & 0x3f]);
        out.push_back(k_alphabet[group >> 12 & 0x3f]);
        out.push_back(k_alphabet[group >> 6 & 0x3f]);
        out.push_back('=');
    }
    return out;
}

std::optional<std::string> base64_decode(std::string_view text) {
    const auto value = [](char c) -> i32 {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string out;
    out.reserve(text.size() / 4 * 3);
    u32 group = 0;
    usize bits = 0;
    for (const char c : text) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        if (c == '=') break;
        const auto v = value(c);
        if (v < 0) return std::nullopt;
        group = group << 6 | static_cast<u32>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>(group >> bits & 0xff));
        }
    }
    return out;
}

namespace {

/// One 64-byte block into the running state.
void sha1_block(std::array<u32, 5> &h, const char *block) {
    std::array<u32, 80> w{};
    for (usize t = 0; t < 16; ++t) {
        const auto *p = reinterpret_cast<const u8 *>(block + t * 4);
        w[t] = static_cast<u32>(p[0]) << 24 | static_cast<u32>(p[1]) << 16 | static_cast<u32>(p[2]) << 8 | static_cast<u32>(p[3]);
    }
    for (usize t = 16; t < 80; ++t) w[t] = std::rotl(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1);

    u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (usize t = 0; t < 80; ++t) {
        u32 f = 0;
        u32 k = 0;
        if (t < 20) {
            f = (b & c) | (~b & d);
            k = 0x5A827999;
        } else if (t < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        } else if (t < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }
        const u32 temp = std::rotl(a, 5) + f + e + k + w[t];
        e = d;
        d = c;
        c = std::rotl(b, 30);
        b = a;
        a = temp;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
}

} // namespace

void Sha1::update(std::string_view bytes) {
    length += bytes.size();
    if (!pending.empty()) {
        const usize take = std::min(bytes.size(), 64 - pending.size());
        pending.append(bytes.substr(0, take));
        bytes.remove_prefix(take);
        if (pending.size() < 64) return;
        sha1_block(state, pending.data());
        pending.clear();
    }
    while (bytes.size() >= 64) {
        sha1_block(state, bytes.data());
        bytes.remove_prefix(64);
    }
    pending.assign(bytes);
}

std::array<u8, 20> Sha1::finish() {
    const u64 bit_length = length * 8;
    std::string tail = std::move(pending);
    tail.push_back(static_cast<char>(0x80));
    while (tail.size() % 64 != 56) tail.push_back('\0');
    for (i32 shift = 56; shift >= 0; shift -= 8) tail.push_back(static_cast<char>(bit_length >> shift & 0xff));
    for (usize block = 0; block < tail.size(); block += 64) sha1_block(state, tail.data() + block);

    std::array<u8, 20> digest{};
    for (usize i = 0; i < 5; ++i) {
        digest[i * 4] = static_cast<u8>(state[i] >> 24);
        digest[i * 4 + 1] = static_cast<u8>(state[i] >> 16);
        digest[i * 4 + 2] = static_cast<u8>(state[i] >> 8);
        digest[i * 4 + 3] = static_cast<u8>(state[i]);
    }
    return digest;
}

std::array<u8, 20> sha1(std::string_view bytes) {
    Sha1 hash;
    hash.update(bytes);
    return hash.finish();
}

} // namespace lilia::net
