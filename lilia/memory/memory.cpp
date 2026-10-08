#include "memory.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>

#include <glaze/json.hpp>

#include <lighter/http/bound_client.h>

#include <lilia/chat/delivery.h>
#include <lilia/log.h>

namespace lilia::memory {

using lighter::Task;

namespace {

constexpr usize k_embed_batch = 16;
constexpr usize k_scan_messages = 2'000;
/// At most this many literal keyword hits add to a chunk's score.
constexpr usize k_keyword_bonus_cap = 2;
constexpr double k_keyword_bonus = 0.05;

struct EmbeddingItem {
    usize index = 0;
    std::vector<float> embedding;
};

struct EmbeddingResponse {
    std::vector<EmbeddingItem> data;
};

struct EmbeddingRequest {
    std::vector<std::string> input;
    std::string model;
};

std::string speaker(const store::StoredMessage &message, std::string_view self_name) {
    if (message.from_self) return std::string(self_name);
    return message.sender_name.empty() ? std::to_string(message.sender_id) : message.sender_name;
}

bool is_cjk(char32_t c) { return (c >= 0x3400 && c <= 0x9fff) || (c >= 0xf900 && c <= 0xfaff); }

bool is_word(char32_t c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

/// Decodes one UTF-8 code point at `i` (advancing it); invalid bytes read as U+FFFD.
char32_t next_code_point(std::string_view text, usize &i) {
    const auto lead = static_cast<unsigned char>(text[i]);
    const usize length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xe ? 3 : (lead >> 3) == 0x1e ? 4 : 0;
    if (length == 0 || i + length > text.size()) {
        ++i;
        return U'�';
    }
    char32_t c = length == 1 ? lead : lead & (0xff >> (length + 1));
    for (usize k = 1; k < length; ++k) c = (c << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3f);
    i += length;
    return c;
}

} // namespace

std::string encode_vector(const std::vector<float> &vector) {
    static_assert(std::endian::native == std::endian::little, "vectors are stored little-endian");
    std::string bytes(vector.size() * sizeof(float), '\0');
    if (!vector.empty()) std::memcpy(bytes.data(), vector.data(), bytes.size());
    return bytes;
}

std::vector<float> decode_vector(std::string_view bytes) {
    std::vector<float> vector(bytes.size() / sizeof(float));
    if (!vector.empty()) std::memcpy(vector.data(), bytes.data(), vector.size() * sizeof(float));
    return vector;
}

double cosine(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.empty() || a.size() != b.size()) return 0;
    double dot = 0, na = 0, nb = 0;
    for (usize i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return na == 0 || nb == 0 ? 0 : dot / (std::sqrt(na) * std::sqrt(nb));
}

std::vector<std::string> keywords(std::string_view query) {
    std::vector<std::string> words;
    std::string current;
    int kind = 0; // 1: CJK run, 2: ASCII word
    usize count = 0;
    const auto flush = [&] {
        if (count >= 2) words.push_back(current);
        current.clear();
        kind = 0;
        count = 0;
    };
    for (usize i = 0; i < query.size();) {
        const usize start = i;
        const char32_t c = next_code_point(query, i);
        const int this_kind = is_cjk(c) ? 1 : is_word(c) ? 2 : 0;
        if (this_kind != kind) flush();
        if (this_kind == 0) continue;
        kind = this_kind;
        current.append(query.substr(start, i - start));
        ++count;
    }
    flush();
    return words;
}

std::vector<Hit> rank(std::vector<store::MemoryChunk> chunks, const std::vector<float> &query, const std::vector<std::string> &words,
                      usize limit, double min_score) {
    std::vector<Hit> hits;
    for (auto &chunk : chunks) {
        double score = cosine(decode_vector(chunk.vector), query);
        // Mostly helps names and Latin words (MC, Godot); meaning-level
        // matches come from the vectors.
        usize literal = 0;
        for (const auto &word : words) {
            if (chunk.text.find(word) != std::string::npos) ++literal;
        }
        score += k_keyword_bonus * static_cast<double>(std::min(literal, k_keyword_bonus_cap));
        if (score >= min_score) hits.push_back({.chunk = std::move(chunk), .score = score});
    }
    std::ranges::sort(hits, [](const Hit &a, const Hit &b) { return a.score > b.score; });
    if (hits.size() > limit) hits.resize(limit);
    return hits;
}

std::vector<std::vector<store::StoredMessage>> complete_chunks(const std::vector<store::StoredMessage> &messages, usize size,
                                                               i64 gap_seconds, i64 now) {
    std::vector<std::vector<store::StoredMessage>> chunks;
    std::vector<store::StoredMessage> current;
    for (const auto &message : messages) {
        if (!current.empty() && message.time - current.back().time >= gap_seconds) {
            chunks.push_back(std::exchange(current, {}));
        }
        current.push_back(message);
        if (current.size() >= std::max<usize>(size, 1)) chunks.push_back(std::exchange(current, {}));
    }
    // A tail that is still growing waits; one the chat has moved on from is done.
    if (!current.empty() && now - current.back().time >= gap_seconds) chunks.push_back(std::move(current));
    return chunks;
}

std::string display_text(const std::vector<store::StoredMessage> &messages, std::string_view self_name, i32 utc_offset_minutes) {
    std::string text;
    for (const auto &message : messages) {
        const auto local = std::chrono::sys_seconds(std::chrono::seconds(message.time + i64{utc_offset_minutes} * 60));
        text += std::format("[{:%m-%d %H:%M}] {}: {}\n", local, speaker(message, self_name), message.text);
    }
    return text;
}

std::string embed_text(const std::vector<store::StoredMessage> &messages, std::string_view self_name) {
    std::string text;
    for (const auto &message : messages) text += std::format("{}: {}\n", speaker(message, self_name), message.text);
    return text;
}

std::string render_hits(const std::vector<Hit> &hits) {
    if (hits.empty()) return "没找到相关的聊天记录。";
    std::string text = "找到的相关聊天记录（按相关程度排序）：\n";
    for (usize i = 0; i < hits.size(); ++i) text += std::format("{}{}", i ? "---\n" : "", hits[i].chunk.text);
    return text;
}

Memory::Memory(const MemoryConfig &config, std::string self_name, i32 utc_offset_minutes, store::Store &store)
    : now([] { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }),
      config(config), self_name(std::move(self_name)), utc_offset_minutes(utc_offset_minutes), store(store) {}

Task<Result<std::vector<std::vector<float>>>> Memory::embed(std::vector<std::string> texts) {
    auto body = glz::write_json(EmbeddingRequest{.input = std::move(texts), .model = config.embedding_model});
    if (!body) co_return lighter::outcome_error(Error::protocol("cannot encode the embedding request"));
    auto request = http.on().post(config.embedding_url + "/v1/embeddings");
    request.json_text(*std::move(body));
    request.timeout(std::chrono::seconds(60));
    auto response = co_await std::move(request).send();
    if (response.has_error()) co_return lighter::outcome_error(Error::protocol("embedding request failed: " + response.error().message()));
    if (!response->ok())
        co_return lighter::outcome_error(Error::protocol("embedding service returned HTTP " + std::to_string(response->status)));
    EmbeddingResponse parsed;
    if (auto ec = glz::read<glz::opts{.null_terminated = false, .error_on_unknown_keys = false}>(parsed, response->text())) {
        co_return lighter::outcome_error(Error::protocol("cannot read the embedding response"));
    }
    std::ranges::sort(parsed.data, {}, &EmbeddingItem::index);
    std::vector<std::vector<float>> vectors;
    for (auto &item : parsed.data) vectors.push_back(std::move(item.embedding));
    co_return vectors;
}

Task<> Memory::index(store::ChatKey chat) {
    if (std::ranges::contains(indexing, chat)) {
        if (!std::ranges::contains(again, chat)) again.push_back(chat);
        co_return;
    }
    indexing.push_back(chat);
    for (bool more = true; more;) {
        more = false;
        auto last = store.last_chunk_row(chat, config.embedding_model);
        if (last.has_error()) {
            log::error("memory: {}", last.error().message());
            break;
        }
        auto messages = store.messages_after(chat, *last, k_scan_messages);
        if (messages.has_error()) {
            log::error("memory: {}", messages.error().message());
            break;
        }
        const auto chunks = complete_chunks(*messages, config.chunk_messages, config.chunk_gap_seconds, now());
        usize stored = 0;
        for (usize start = 0; start < chunks.size(); start += k_embed_batch) {
            const usize end = std::min(chunks.size(), start + k_embed_batch);
            std::vector<std::string> texts;
            for (usize i = start; i < end; ++i) texts.push_back(embed_text(chunks[i], self_name));
            auto vectors = co_await embed(std::move(texts));
            if (vectors.has_error() || vectors->size() != end - start) {
                log::warn("memory: embedding failed for {}:{}: {}", chat.type, chat.id,
                          vectors.has_error() ? vectors.error().message() : "wrong number of vectors");
                break;
            }
            for (usize i = start; i < end; ++i) {
                const auto &group = chunks[i];
                auto added = store.add_chunk({
                    .chat = chat,
                    .first_row = group.front().row_id,
                    .last_row = group.back().row_id,
                    .start_time = group.front().time,
                    .end_time = group.back().time,
                    .text = display_text(group, self_name, utc_offset_minutes),
                    .model = config.embedding_model,
                    .vector = encode_vector((*vectors)[i - start]),
                });
                if (added.has_error()) log::error("memory: {}", added.error().message());
                ++stored;
            }
        }
        if (stored > 0) log::info("memory: indexed {} chunk(s) of {}:{}", stored, chat.type, chat.id);
        // A full scan means more history is waiting (the first backfill).
        more = messages->size() == k_scan_messages && stored > 0;
        if (const auto it = std::ranges::find(again, chat); it != again.end()) {
            again.erase(it);
            more = true;
        }
    }
    std::erase(indexing, chat);
}

Task<std::vector<Hit>> Memory::search(store::ChatKey chat, std::string query) {
    const auto prefixed =
        config.query_instruction.empty() ? query : std::format("Instruct: {}\nQuery: {}", config.query_instruction, query);
    auto vectors = co_await embed({prefixed});
    if (vectors.has_error() || vectors->empty()) {
        log::warn("memory: cannot embed the query: {}", vectors.has_error() ? vectors.error().message() : "no vector");
        co_return std::vector<Hit>{};
    }
    auto chunks = store.chunks(chat, config.embedding_model);
    if (chunks.has_error()) {
        log::error("memory: {}", chunks.error().message());
        co_return std::vector<Hit>{};
    }
    auto hits = rank(*std::move(chunks), vectors->front(), keywords(query), config.search_results, config.min_score);
    log::info("memory: search \"{}\" in {}:{} -> {} hit(s){}", query, chat.type, chat.id, hits.size(),
              hits.empty() ? "" : std::format(", best {:.2f}", hits.front().score));
    co_return hits;
}

} // namespace lilia::memory
