#include "retriever.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <unordered_set>

#include <glaze/json.hpp>

#include <lighter/http/bound_client.h>

#include <lilia/log.h>
#include <lilia/net/codec.h>
#include <lilia/retrieval/lexical.h>

namespace lilia::retrieval {

using lighter::Task;

namespace {

/// Texts per embedding request; queries kept; texts per backfill step.
constexpr usize k_batch = 32;
constexpr usize k_kept_queries = 64;
constexpr usize k_backfill_step = 256;
/// Bumped when what is embedded for a corpus changes shape.
constexpr std::string_view k_format_version = "v1";

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

} // namespace

std::string encode(const Vector &vector) {
    static_assert(std::endian::native == std::endian::little, "vectors are stored little-endian");
    std::string bytes(vector.size() * sizeof(float), '\0');
    if (!vector.empty()) std::memcpy(bytes.data(), vector.data(), bytes.size());
    return bytes;
}

Vector decode(std::string_view bytes) {
    Vector vector(bytes.size() / sizeof(float));
    if (!vector.empty()) std::memcpy(vector.data(), bytes.data(), vector.size() * sizeof(float));
    return vector;
}

void normalize(Vector &vector) {
    double norm = 0;
    for (const float x : vector) norm += static_cast<double>(x) * x;
    if (norm == 0) return;
    const auto scale = static_cast<float>(1.0 / std::sqrt(norm));
    for (float &x : vector) x *= scale;
}

double dot(const Vector &a, const Vector &b) {
    if (a.empty() || a.size() != b.size()) return 0;
    double sum = 0;
    for (usize i = 0; i < a.size(); ++i) sum += static_cast<double>(a[i]) * b[i];
    return sum;
}

std::string profile_of(const MemoryConfig &config) { return std::format("{}/{}", config.embedding_model, k_format_version); }

std::string text_hash(std::string_view text) {
    static constexpr std::string_view k_hex = "0123456789abcdef";
    std::string out;
    for (const u8 byte : net::sha1(text)) {
        out.push_back(k_hex[byte >> 4]);
        out.push_back(k_hex[byte & 0xf]);
    }
    return out;
}

std::string query_text(std::string_view task, std::string_view text) {
    return task.empty() ? std::string(text) : std::format("Instruct: {}\nQuery:{}", task, text);
}

std::vector<Match> fuse(const std::vector<std::optional<double>> &dense, const std::vector<double> &boosts,
                        const std::vector<double> &lexical, const Options &options, const std::vector<const Vector *> &vectors) {
    const usize count = dense.size();
    const auto passes = [&](usize i) { return !dense[i] || *dense[i] >= options.min_similarity; };
    // MMR picks from everything that passes; otherwise each leg ranks a few
    // times what is wanted.
    const usize pool = options.mmr_lambda > 0 ? count : options.top_k * k_leg_factor;
    const auto boost = [&](usize i) { return i < boosts.size() ? boosts[i] : 0.0; };

    std::vector<usize> by_meaning;
    for (usize i = 0; i < count; ++i) {
        if (dense[i] && passes(i)) by_meaning.push_back(i);
    }
    std::ranges::stable_sort(by_meaning, [&](usize a, usize b) { return *dense[a] + boost(a) > *dense[b] + boost(b); });
    if (by_meaning.size() > pool) by_meaning.resize(pool);

    std::vector<usize> by_words;
    for (usize i = 0; i < lexical.size() && i < count; ++i) {
        if (lexical[i] > 0 && passes(i)) by_words.push_back(i);
    }
    std::ranges::stable_sort(by_words, [&](usize a, usize b) { return lexical[a] > lexical[b]; });
    if (by_words.size() > pool) by_words.resize(pool);

    std::vector<double> scores(count, 0.0);
    std::vector<bool> found(count, false);
    for (usize rank = 0; rank < by_meaning.size(); ++rank) {
        scores[by_meaning[rank]] += k_dense_weight / (k_rrf_k + static_cast<double>(rank + 1));
        found[by_meaning[rank]] = true;
    }
    for (usize rank = 0; rank < by_words.size(); ++rank) {
        scores[by_words[rank]] += k_lexical_weight / (k_rrf_k + static_cast<double>(rank + 1));
        found[by_words[rank]] = true;
    }
    std::vector<Match> candidates;
    for (usize i = 0; i < count; ++i) {
        if (found[i]) candidates.push_back({.index = i, .similarity = dense[i], .score = scores[i]});
    }
    std::ranges::stable_sort(candidates, [](const Match &a, const Match &b) {
        if (a.score != b.score) return a.score > b.score;
        return a.similarity.value_or(-1) > b.similarity.value_or(-1);
    });
    if (options.mmr_lambda <= 0) {
        if (candidates.size() > options.top_k) candidates.resize(options.top_k);
        return candidates;
    }

    // MMR: relevance against likeness to what is already picked.
    const auto vector_of = [&](usize i) { return i < vectors.size() ? vectors[i] : nullptr; };
    std::vector<Match> picked;
    std::vector<bool> taken(candidates.size(), false);
    while (picked.size() < options.top_k && picked.size() < candidates.size()) {
        usize best = candidates.size();
        double best_value = -1e9;
        for (usize c = 0; c < candidates.size(); ++c) {
            if (taken[c]) continue;
            double redundancy = 0;
            if (const auto *own = vector_of(candidates[c].index)) {
                for (const auto &chosen : picked) {
                    if (const auto *other = vector_of(chosen.index)) redundancy = std::max(redundancy, dot(*own, *other));
                }
            }
            const double value = options.mmr_lambda * candidates[c].similarity.value_or(0) - (1 - options.mmr_lambda) * redundancy;
            if (value > best_value) {
                best_value = value;
                best = c;
            }
        }
        if (best == candidates.size()) break;
        taken[best] = true;
        picked.push_back(candidates[best]);
    }
    return picked;
}

Retriever::Retriever(const MemoryConfig &config, store::Store &store) : config(config), store(store), profile(profile_of(config)) {
    backend = [this](std::vector<std::string> texts) -> Task<Result<std::vector<Vector>>> {
        auto body = glz::write_json(EmbeddingRequest{.input = std::move(texts), .model = this->config.embedding_model});
        if (!body) co_return lighter::outcome_error(Error::protocol("cannot encode the embedding request"));
        auto request = http.on().post(this->config.embedding_url + this->config.embedding_path);
        request.json_text(*std::move(body));
        request.timeout(std::chrono::seconds(60));
        auto response = co_await std::move(request).send();
        if (response.has_error())
            co_return lighter::outcome_error(Error::protocol("embedding request failed: " + response.error().message()));
        if (!response->ok())
            co_return lighter::outcome_error(Error::protocol("embedding service returned HTTP " + std::to_string(response->status)));
        EmbeddingResponse parsed;
        if (auto ec = glz::read<glz::opts{.null_terminated = false, .error_on_unknown_keys = false}>(parsed, response->text())) {
            co_return lighter::outcome_error(Error::protocol("cannot read the embedding response"));
        }
        std::ranges::sort(parsed.data, {}, &EmbeddingItem::index);
        std::vector<Vector> vectors;
        for (auto &item : parsed.data) vectors.push_back(std::move(item.embedding));
        co_return vectors;
    };
}

Task<std::vector<const Vector *>> Retriever::vectors(std::vector<std::string> texts) {
    std::vector<std::string> hashes;
    hashes.reserve(texts.size());
    for (const auto &text : texts) hashes.push_back(text.empty() ? std::string() : text_hash(text));

    // Not in memory: look in the store, then ask the service.
    std::vector<std::string> unknown;
    std::vector<usize> unknown_text;
    std::unordered_set<std::string_view> listed;
    for (usize i = 0; i < texts.size(); ++i) {
        if (hashes[i].empty() || cache.contains(hashes[i]) || !listed.insert(hashes[i]).second) continue;
        unknown.push_back(hashes[i]);
        unknown_text.push_back(i);
    }
    if (!unknown.empty()) {
        if (auto stored = store.embeddings(profile, unknown); stored.has_value()) {
            for (auto &[hash, bytes] : *stored) cache.emplace(hash, decode(bytes));
        } else {
            log::error("retrieval: {}", stored.error().message());
        }
    }
    std::vector<usize> missing;
    for (usize k = 0; k < unknown.size(); ++k) {
        if (!cache.contains(unknown[k])) missing.push_back(unknown_text[k]);
    }
    for (usize start = 0; start < missing.size(); start += k_batch) {
        const usize end = std::min(missing.size(), start + k_batch);
        std::vector<std::string> batch;
        for (usize k = start; k < end; ++k) batch.push_back(texts[missing[k]]);
        auto embedded = co_await backend(std::move(batch));
        if (embedded.has_error() || embedded->size() != end - start) {
            if (!failing) {
                log::warn("retrieval: embedding failed: {}", embedded.has_error() ? embedded.error().message() : "wrong number of vectors");
            }
            failing = true;
            break;
        }
        if (failing) log::info("retrieval: the embedding service answers again");
        failing = false;
        std::vector<std::pair<std::string, std::string>> rows;
        for (usize k = start; k < end; ++k) {
            auto &vector = (*embedded)[k - start];
            normalize(vector);
            rows.emplace_back(hashes[missing[k]], encode(vector));
            cache.insert_or_assign(hashes[missing[k]], std::move(vector));
        }
        if (auto added = store.add_embeddings(profile, rows); added.has_error()) log::error("retrieval: {}", added.error().message());
    }

    std::vector<const Vector *> out;
    out.reserve(texts.size());
    for (const auto &hash : hashes) {
        const auto found = hash.empty() ? cache.end() : cache.find(hash);
        out.push_back(found == cache.end() ? nullptr : &found->second);
    }
    co_return out;
}

Task<std::optional<Vector>> Retriever::query(std::string_view task, std::string text) {
    auto key = query_text(task, text);
    if (const auto found = queries.find(key); found != queries.end()) co_return found->second;
    auto embedded = co_await backend({key});
    if (embedded.has_error() || embedded->size() != 1) {
        if (!failing) log::warn("retrieval: cannot embed a query: {}", embedded.has_error() ? embedded.error().message() : "no vector");
        failing = true;
        co_return std::nullopt;
    }
    failing = false;
    auto vector = std::move(embedded->front());
    normalize(vector);
    queries.insert_or_assign(key, vector);
    query_order.push_back(std::move(key));
    if (query_order.size() > k_kept_queries) {
        queries.erase(query_order.front());
        query_order.pop_front();
    }
    co_return vector;
}

Task<usize> Retriever::backfill(std::vector<std::string> texts) {
    usize missing = 0;
    for (usize start = 0; start < texts.size(); start += k_backfill_step) {
        const usize end = std::min(texts.size(), start + k_backfill_step);
        const auto found =
            co_await vectors({texts.begin() + static_cast<std::ptrdiff_t>(start), texts.begin() + static_cast<std::ptrdiff_t>(end)});
        for (usize i = 0; i < found.size(); ++i) missing += found[i] == nullptr && !texts[start + i].empty();
    }
    co_return missing;
}

const std::vector<std::string> &Retriever::terms_of(const std::string &text) {
    const auto hash = text_hash(text);
    if (const auto found = lexicon.find(hash); found != lexicon.end()) return found->second;
    return lexicon.emplace(hash, terms(text)).first->second;
}

Task<std::vector<Match>> Retriever::search(std::string_view task, std::string query, const std::vector<Document> &documents,
                                           Options options) {
    std::vector<std::optional<double>> dense(documents.size());
    std::vector<const Vector *> firsts(documents.size(), nullptr);
    if (const auto asked = co_await this->query(task, query)) {
        std::vector<std::string> texts;
        for (const auto &document : documents) texts.insert(texts.end(), document.texts.begin(), document.texts.end());
        const auto found = co_await vectors(std::move(texts));
        usize at = 0;
        for (usize i = 0; i < documents.size(); ++i) {
            for (usize k = 0; k < documents[i].texts.size(); ++k, ++at) {
                if (!found[at]) continue;
                if (!firsts[i]) firsts[i] = found[at];
                const double similarity = dot(*asked, *found[at]);
                dense[i] = dense[i] ? std::max(*dense[i], similarity) : similarity;
            }
        }
    }
    std::vector<double> lexical;
    if (options.lexical) {
        static const std::vector<std::string> k_none;
        std::vector<const std::vector<std::string> *> indexed;
        for (const auto &document : documents) indexed.push_back(document.lexical.empty() ? &k_none : &terms_of(document.lexical));
        lexical = bm25(terms(query), indexed);
    }
    std::vector<double> boosts;
    for (const auto &document : documents) boosts.push_back(document.boost);
    co_return fuse(dense, boosts, lexical, options, firsts);
}

Task<std::optional<std::vector<double>>> Retriever::similarities(std::string text, std::vector<std::string> others) {
    others.insert(others.begin(), std::move(text));
    const auto found = co_await vectors(std::move(others));
    if (!found.front()) co_return std::nullopt;
    std::vector<double> out;
    for (usize i = 1; i < found.size(); ++i) out.push_back(found[i] ? dot(*found.front(), *found[i]) : 0.0);
    co_return out;
}

} // namespace lilia::retrieval
