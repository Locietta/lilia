#include "memory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>

#include <lilia/chat/delivery.h>
#include <lilia/log.h>

namespace lilia::memory {

using lighter::Task;

namespace {

constexpr usize k_scan_messages = 2'000;
/// Up to this much for the newest chunk, fading with this time constant.
constexpr double k_recency_bonus = 0.04;
constexpr double k_recency_seconds = 14.0 * 86400;

std::string speaker(const store::StoredMessage &message, std::string_view self_name) {
    if (message.from_self) return std::string(self_name);
    return message.sender_name.empty() ? std::to_string(message.sender_id) : message.sender_name;
}

} // namespace

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

Memory::Memory(const MemoryConfig &config, std::string self_name, i32 utc_offset_minutes, store::Store &store,
               retrieval::Retriever &retriever)
    : now([] { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }),
      config(config), self_name(std::move(self_name)), utc_offset_minutes(utc_offset_minutes), store(store), retriever(retriever) {}

std::vector<std::string> Memory::embedded_texts() {
    std::vector<std::string> texts;
    if (auto chunks = store.search_texts(); chunks.has_value()) texts = *std::move(chunks);
    if (auto summaries = store.all_summaries(); summaries.has_value()) {
        for (const auto &summary : *summaries) {
            for (auto &text : summary_texts(summary)) texts.push_back(std::move(text));
        }
    }
    if (auto facts = store.all_facts(); facts.has_value()) {
        for (auto &fact : *facts) texts.push_back(std::move(fact.fact));
    }
    return texts;
}

Task<> Memory::index(store::ChatKey chat) {
    if (std::ranges::contains(indexing, chat)) {
        if (!std::ranges::contains(again, chat)) again.push_back(chat);
        co_return;
    }
    indexing.push_back(chat);
    for (bool more = true; more;) {
        more = false;
        auto last = store.last_chunk_row(chat);
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
        std::vector<std::string> texts;
        for (const auto &group : chunks) {
            store::MemoryChunk chunk{
                .chat = chat,
                .first_row = group.front().row_id,
                .last_row = group.back().row_id,
                .start_time = group.front().time,
                .end_time = group.back().time,
                .text = display_text(group, self_name, utc_offset_minutes),
                .search_text = embed_text(group, self_name),
            };
            if (auto added = store.add_chunk(chunk); added.has_error()) {
                log::error("memory: {}", added.error().message());
                break;
            }
            texts.push_back(std::move(chunk.search_text));
        }
        if (!texts.empty()) {
            // Embedded now so a search does not wait; one that fails is
            // embedded by the next search or backfill.
            const usize stored = texts.size();
            const auto missing = co_await retriever.backfill(std::move(texts));
            log::info("memory: indexed {} chunk(s) of {}:{}{}", stored, chat.type, chat.id,
                      missing ? std::format(" ({} not embedded yet)", missing) : "");
        }
        co_await summarize(chat);
        // A full scan means more history is waiting (the first backfill).
        more = messages->size() == k_scan_messages && !chunks.empty();
        if (const auto it = std::ranges::find(again, chat); it != again.end()) {
            again.erase(it);
            more = true;
        }
    }
    std::erase(indexing, chat);
}

Task<std::vector<Hit>> Memory::search(store::ChatKey chat, std::string query) {
    auto chunks = store.chunks(chat);
    if (chunks.has_error()) {
        log::error("memory: {}", chunks.error().message());
        co_return std::vector<Hit>{};
    }
    // Newer memories win ties: a small bonus fading over about two weeks.
    const i64 current = now();
    const auto recency = [&](i64 end_time) {
        return k_recency_bonus * std::exp(-static_cast<double>(std::max<i64>(0, current - end_time)) / k_recency_seconds);
    };
    std::vector<retrieval::Document> documents;
    for (const auto &chunk : *chunks) {
        documents.push_back({.id = chunk.id, .texts = {chunk.search_text}, .lexical = chunk.search_text, .boost = recency(chunk.end_time)});
    }
    // Summaries compete with raw chunks; they cover more ground per hit.
    if (auto summaries = store.summaries(chat, 0); summaries.has_value()) {
        for (const auto &summary : *summaries) {
            if (summary.summary.empty()) continue;
            auto texts = summary_texts(summary);
            std::string lexical;
            for (const auto &text : texts) lexical += text + "\n";
            documents.push_back(
                {.id = -summary.id, .texts = std::move(texts), .lexical = std::move(lexical), .boost = recency(summary.end_time)});
            chunks->push_back(summary_chunk(summary, utc_offset_minutes));
        }
    }
    const auto matches = co_await retriever.search(config.query_instruction, query, documents,
                                                   {.top_k = config.search_results, .min_similarity = config.min_score, .lexical = true});
    std::vector<Hit> hits;
    for (const auto &match : matches) hits.push_back({.chunk = (*chunks)[match.index], .score = match.similarity.value_or(0)});
    log::info("memory: search \"{}\" in {}:{} -> {} hit(s){}", query, chat.type, chat.id, hits.size(),
              hits.empty() ? "" : std::format(", best {:.2f}", hits.front().score));
    co_return hits;
}

} // namespace lilia::memory
