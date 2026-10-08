#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <lighter/async/runtime/task.h>
#include <lighter/http/client.h>

#include <lilia/config.h>
#include <lilia/error.h>
#include <lilia/store/store.h>

namespace lilia::memory {

// Long-term memory, part one: searchable history of the allowed chats.
// Consecutive messages are grouped into small chunks, embedded by a local
// model (deployment/embedding) and stored with the model's name; a search
// ranks a chat's chunks by cosine similarity, with a small bonus for literal
// keyword hits. Chunks of another model are dropped and rebuilt.

struct Hit {
    store::MemoryChunk chunk;
    double score = 0;
};

/// Raw little-endian float32, as stored in memory_chunks.vector.
std::string encode_vector(const std::vector<float> &vector);
std::vector<float> decode_vector(std::string_view bytes);

/// Cosine similarity; 0 when either side is empty or the sizes differ.
double cosine(const std::vector<float> &a, const std::vector<float> &b);

/// Words for the literal bonus: runs of CJK (2+ characters) or ASCII letters
/// and digits (2+), split at spaces and punctuation.
std::vector<std::string> keywords(std::string_view query);

/// The best `limit` chunks scoring at least `min_score`, best first.
std::vector<Hit> rank(std::vector<store::MemoryChunk> chunks, const std::vector<float> &query, const std::vector<std::string> &words,
                      usize limit, double min_score);

/// Splits messages (oldest first) into chunks of at most `size`, starting a
/// new chunk at gaps of `gap_seconds`. The last group is complete only when
/// it is full or older than `gap_seconds` at `now`; incomplete groups are
/// left for later and not returned.
std::vector<std::vector<store::StoredMessage>> complete_chunks(const std::vector<store::StoredMessage> &messages, usize size,
                                                               i64 gap_seconds, i64 now);

/// "[10-05 21:03] 名字: 内容" lines, the stored and displayed form.
std::string display_text(const std::vector<store::StoredMessage> &messages, std::string_view self_name, i32 utc_offset_minutes);

/// "名字: 内容" lines without times, the embedded form.
std::string embed_text(const std::vector<store::StoredMessage> &messages, std::string_view self_name);

/// Search results as the planner and the writer see them.
std::string render_hits(const std::vector<Hit> &hits);

struct Memory {
    Memory(const MemoryConfig &config, std::string self_name, i32 utc_offset_minutes, store::Store &store);

    /// Embeds `texts` through the OpenAI-compatible /v1/embeddings endpoint.
    lighter::Task<Result<std::vector<std::vector<float>>>> embed(std::vector<std::string> texts);

    /// Chunks and embeds the messages of `chat` not covered yet (the whole
    /// history the first time). Calls for a chat while one runs are merged.
    lighter::Task<> index(store::ChatKey chat);

    /// The best chunks of `chat` for `query`.
    lighter::Task<std::vector<Hit>> search(store::ChatKey chat, std::string query);

    std::move_only_function<i64()> now;
    const MemoryConfig &config;
    std::string self_name;
    i32 utc_offset_minutes;
    store::Store &store;
    lighter::http::Client http;
    std::vector<store::ChatKey> indexing; ///< chats with an index() running
    std::vector<store::ChatKey> again;    ///< asked again while running
};

} // namespace lilia::memory
