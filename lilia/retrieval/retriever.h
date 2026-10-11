#pragma once

#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <lighter/async/runtime/task.h>
#include <lighter/http/client.h>

#include <lilia/config.h>
#include <lilia/error.h>
#include <lilia/store/store.h>

namespace lilia::retrieval {

// Retrieval for everything lilia embeds (chat history, summaries, facts,
// expressions, stickers), after what MaiBot, OpenClaw and LangChain do:
//
// - One embedding cache, keyed by profile (the model and this input format)
//   and the text's hash, in memory and in the `embeddings` table. A text is
//   embedded once; another model gets a profile of its own, and the startup
//   backfill re-embeds everything under it.
// - Queries carry Qwen3-Embedding's instruction ("Instruct: …\nQuery:…"),
//   one task per corpus; documents are embedded bare. Vectors are
//   L2-normalized, so similarity is a dot product.
// - Two legs: dense (brute force; the corpora are thousands of rows at most)
//   and, where asked, BM25 keywords (lexical.h). Each leg ranks
//   `k_leg_factor` x top_k; reciprocal rank fusion (k 60, weights 0.7/0.3)
//   merges them. The relevance gate looks at the dense similarity only.
// - Optional MMR among the fused candidates, for variety (expressions).
//
// The callers choose the candidates (a chat, a person, a status); the
// retriever only ranks them.

using Vector = std::vector<float>;

/// Query tasks, in English as Qwen3-Embedding recommends. Chat search uses
/// memory.query_instruction.
inline constexpr std::string_view k_recall_task =
    "Given the latest group chat messages, retrieve an earlier conversation summary about the same topic";
inline constexpr std::string_view k_self_task =
    "Given recent group chat messages, retrieve things the speaker has said about herself that are relevant to them";
inline constexpr std::string_view k_expression_task = "Given a chat situation and what to say, retrieve ways of speaking that would fit it";
inline constexpr std::string_view k_sticker_task = "Given a feeling or reaction, retrieve stickers whose tags and description express it";

/// Raw little-endian float32, as stored.
std::string encode(const Vector &vector);
Vector decode(std::string_view bytes);

/// Scales `vector` to unit length (a zero vector stays as it is).
void normalize(Vector &vector);

/// Dot product; 0 when either is empty or the sizes differ.
double dot(const Vector &a, const Vector &b);

/// The embedding cache's profile for this configuration: the model and the
/// input format version ("Qwen3-Embedding-0.6B/v1").
std::string profile_of(const MemoryConfig &config);

/// Hex SHA-1 of `text`: its key in the embedding cache.
std::string text_hash(std::string_view text);

/// Qwen3-Embedding's query form, "Instruct: {task}\nQuery:{text}"; `text`
/// alone when `task` is empty.
std::string query_text(std::string_view task, std::string_view text);

/// One candidate. Its similarity is the best of its `texts` (a summary and
/// its recall cues); `lexical` is what the keyword leg reads (empty: none).
struct Document {
    i64 id = 0;
    std::vector<std::string> texts;
    std::string lexical;
    /// Added to the similarity when ranking, never when gating (recency).
    double boost = 0;
};

struct Options {
    usize top_k = 5;
    /// The least dense similarity a result may have. A document without a
    /// vector (the embedding service failed) can only come in as a keyword hit.
    double min_similarity = 0;
    bool lexical = false;
    /// MMR's weight on relevance against variety; 0 turns MMR off.
    double mmr_lambda = 0;
};

struct Match {
    usize index = 0;                  ///< into the documents searched
    std::optional<double> similarity; ///< dense; nullopt without a vector
    double score = 0;                 ///< fused
};

/// What both legs found, fused: `dense[i]` and `lexical[i]` score document
/// i (nullopt / 0: not found by that leg). Ranks and gates as described
/// above; MMR, if asked, needs `vectors` (the first vector of each document).
std::vector<Match> fuse(const std::vector<std::optional<double>> &dense, const std::vector<double> &boosts,
                        const std::vector<double> &lexical, const Options &options, const std::vector<const Vector *> &vectors = {});

/// RRF constants and how many each leg ranks per wanted result.
inline constexpr double k_rrf_k = 60;
inline constexpr double k_dense_weight = 0.7;
inline constexpr double k_lexical_weight = 0.3;
inline constexpr usize k_leg_factor = 4;

struct Retriever {
    /// One request to the embedding service: a vector per text, in order.
    using Backend = std::move_only_function<lighter::Task<Result<std::vector<Vector>>>(std::vector<std::string>)>;

    /// Embeds through memory.embedding_url + embedding_path.
    Retriever(const MemoryConfig &config, store::Store &store);

    /// A vector for each text, null where it could not be embedded; from
    /// memory, then the store, then the service (in batches, then stored).
    /// The pointers stay valid for the retriever's life.
    lighter::Task<std::vector<const Vector *>> vectors(std::vector<std::string> texts);

    /// `text` as a query for `task`; recent queries are kept.
    lighter::Task<std::optional<Vector>> query(std::string_view task, std::string text);

    /// Embeds whichever of `texts` are not cached yet. Returns how many
    /// still have no vector.
    lighter::Task<usize> backfill(std::vector<std::string> texts);

    /// The best of `documents` for `query` under `task`, best first.
    lighter::Task<std::vector<Match>> search(std::string_view task, std::string query, const std::vector<Document> &documents,
                                             Options options);

    /// How alike `text` is to each of `others` (both as documents); nullopt
    /// when they could not be embedded.
    lighter::Task<std::optional<std::vector<double>>> similarities(std::string text, std::vector<std::string> others);

    /// Index terms of `text` (lexical.h), cached.
    const std::vector<std::string> &terms_of(const std::string &text);

    const MemoryConfig &config;
    store::Store &store;
    Backend backend;
    /// The cache's profile: the model and the input format version.
    std::string profile;
    lighter::http::Client http;
    std::unordered_map<std::string, Vector> cache;                     ///< by text hash
    std::unordered_map<std::string, std::vector<std::string>> lexicon; ///< terms by text hash
    std::unordered_map<std::string, Vector> queries;                   ///< recent queries
    std::deque<std::string> query_order;                               ///< oldest first
    bool failing = false;                                              ///< the last request failed (logged once)
};

} // namespace lilia::retrieval
