#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <lighter/async/io/loop.h>

#include <lilia/retrieval/lexical.h>
#include <lilia/retrieval/retriever.h>
#include <lilia/store/store.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;
using lighter::Task;

/// A fake embedding service: a direction per known word (unknown text points
/// elsewhere), counting what it is asked to embed.
struct FakeService {
    std::map<std::string, retrieval::Vector> words{
        {"芒果", {1, 0, 0}},
        {"MC", {0, 1, 0}},
        {"无语", {0, 0, 1}},
    };
    usize requests = 0;
    usize texts = 0;
    bool down = false;
    std::vector<std::string> last_batch;
    std::string last_instruction;

    retrieval::Retriever::Backend backend() {
        return [this](std::vector<std::string> batch, std::string instruction) -> Task<Result<std::vector<retrieval::Vector>>> {
            ++requests;
            texts += batch.size();
            last_batch = batch;
            last_instruction = std::move(instruction);
            if (down) co_return lighter::outcome_error(Error::protocol("down"));
            std::vector<retrieval::Vector> out;
            for (const auto &text : batch) {
                retrieval::Vector vector{0.05f, 0.05f, 0.05f};
                for (const auto &[word, direction] : words) {
                    if (text.find(word) == std::string::npos) continue;
                    for (usize i = 0; i < 3; ++i) vector[i] += 2 * direction[i]; // unnormalized on purpose
                }
                out.push_back(std::move(vector));
            }
            co_return out;
        };
    }
};

void test_vectors_and_text() {
    const retrieval::Vector v{0.5f, -1.25f, 3.0f};
    require(retrieval::decode(retrieval::encode(v)) == v && retrieval::encode(v).size() == 12, "vectors round-trip as float32 bytes");
    auto unit = retrieval::Vector{3, 4};
    retrieval::normalize(unit);
    require(std::abs(unit[0] - 0.6f) < 1e-6 && std::abs(retrieval::dot(unit, unit) - 1) < 1e-6, "normalized to unit length");
    require(retrieval::dot({1, 0}, {1, 0, 0}) == 0 && retrieval::dot({}, {}) == 0, "mismatched or empty vectors score 0");
    require(retrieval::query_text("Find it", "芒果") == "Instruct: Find it\nQuery:芒果" && retrieval::query_text("", "芒果") == "芒果",
            "Qwen3's query form");
    require(retrieval::text_hash("abc") == "a9993e364706816aba3e25717850c26c9cd0d89d", "SHA-1 of the text");
    require(retrieval::profile_of({.embedding_model = "Qwen3-Embedding-0.6B"}) == "Qwen3-Embedding-0.6B/v1", "profile");
}

void test_lexical() {
    const auto words = retrieval::terms("周六一起打 MC 吗，Godot4 也行");
    const auto has = [&](std::string_view word) { return std::ranges::contains(words, word); };
    require(has("周六") && has("一起") && has("mc") && has("godot4") && !has("吗") && !has("打"),
            "jieba words of two or more characters and lower-cased ASCII words");
    require(has("起打") && has("周六") && std::ranges::count(words, std::string("周六")) == 2, "plus every Han pair");

    const std::vector<std::string> a{"芒果", "过敏"}, b{"周六", "mc", "一起"}, c{"芒果", "芒果", "冰淇淋", "好吃", "夏天"};
    const auto scores = retrieval::bm25({"芒果", "过敏"}, {&a, &b, &c});
    require(scores[0] > scores[2] && scores[2] > 0 && scores[1] == 0, "rarer and denser matches score higher; no match is 0");
    require(retrieval::bm25({}, {&a})[0] == 0 && retrieval::bm25({"x"}, {}).empty(), "nothing to match");
}

void test_fusion() {
    // 0 and 1 by meaning, 2 by words, 3 below the gate, 4 found by neither.
    const std::vector<std::optional<double>> dense{0.9, 0.8, 0.5, 0.2, std::nullopt};
    const std::vector<double> lexical{0, 0, 3.0, 5.0, 0};
    auto matches = retrieval::fuse(dense, {}, lexical, {.top_k = 3, .min_similarity = 0.3});
    require(matches.size() == 3 && matches[0].index == 2 && matches[1].index == 0 && matches[2].index == 1,
            "RRF: found by both legs ranks first; the gate drops a keyword hit that means something else");
    matches = retrieval::fuse({std::nullopt, std::nullopt}, {}, {0, 2.0}, {.top_k = 5, .min_similarity = 0.3});
    require(matches.size() == 1 && matches[0].index == 1 && !matches[0].similarity, "without vectors, keyword hits still count");
    matches = retrieval::fuse({0.80, 0.79}, {0, 0.05}, {}, {.top_k = 2});
    require(matches[0].index == 1 && *matches[0].similarity == 0.79, "a boost reorders, the similarity is reported as it is");
    matches = retrieval::fuse({0.80, 0.79}, {0, 0.05}, {}, {.top_k = 2, .min_similarity = 0.795});
    require(matches.size() == 1 && matches[0].index == 0, "but never lets a document past the gate");

    // MMR: the near twin of the best gives way to something different.
    const retrieval::Vector x{1, 0}, twin{0.995f, 0.0998f}, y{0, 1};
    matches = retrieval::fuse({0.9, 0.89, 0.6}, {}, {}, {.top_k = 2, .mmr_lambda = 0.5}, {&x, &twin, &y});
    require(matches.size() == 2 && matches[0].index == 0 && matches[1].index == 2, "MMR picks for variety");
}

Task<> test_cache() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    const MemoryConfig config{.embedding_model = "fake"};
    FakeService service;
    {
        retrieval::Retriever retriever(config, *db);
        retriever.backend = service.backend();
        auto found = co_await retriever.vectors({"芒果", "MC", "芒果", ""});
        require(found.size() == 4 && found[0] && found[1] && found[0] == found[2] && !found[3], "one vector per text; empty has none");
        require(service.requests == 1 && service.texts == 2, "each text embedded once, in one batch");
        require(std::abs(retrieval::dot(*found[0], *found[0]) - 1) < 1e-6, "stored normalized");
        std::ignore = co_await retriever.vectors({"芒果"});
        require(service.requests == 1, "then served from memory");
        require(co_await retriever.backfill({"芒果", "无语", "新的"}) == 0 && service.texts == 4, "backfill embeds only what is missing");
    }
    {
        retrieval::Retriever again(config, *db);
        again.backend = service.backend();
        std::ignore = co_await again.vectors({"芒果", "MC", "无语"});
        require(service.requests == 2, "a new process reads them from the store");
    }
    {
        const MemoryConfig another{.embedding_model = "another"};
        retrieval::Retriever other(another, *db);
        other.backend = service.backend();
        std::ignore = co_await other.vectors({"芒果"});
        require(service.requests == 3, "another model has its own profile");
        require(*db->delete_embeddings_except(other.profile) == 4 &&
                    db->embeddings(other.profile, {retrieval::text_hash("芒果")})->size() == 1,
                "old profiles can be dropped");
    }
    {
        retrieval::Retriever failing(config, *db);
        service.down = true;
        failing.backend = service.backend();
        const auto found = co_await failing.vectors({"没见过的"});
        require(!found[0] && co_await failing.backfill({"没见过的"}) == 1, "a failure leaves the text without a vector");
        require(!co_await failing.query("task", "芒果"), "and a query without one");
        service.down = false;
    }
}

Task<> test_search() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    FakeService service;
    const MemoryConfig config{.embedding_model = "fake"}, offline_config{.embedding_model = "offline"};
    retrieval::Retriever retriever(config, *db);
    retriever.backend = service.backend();
    const std::vector<retrieval::Document> documents{
        {.id = 10, .texts = {"我对芒果过敏"}, .lexical = "我对芒果过敏"},
        {.id = 11, .texts = {"周六一起打 MC"}, .lexical = "周六一起打 MC"},
        {.id = 12, .texts = {"早", "无语"}, .lexical = "早"},
    };
    auto matches = co_await retriever.search("task", "芒果", documents, {.top_k = 1, .min_similarity = 0.5, .lexical = true});
    require(matches.size() == 1 && documents[matches[0].index].id == 10 && *matches[0].similarity > 0.9, "the closest by meaning");
    matches = co_await retriever.search("task", "无语", documents, {.top_k = 3, .min_similarity = 0.5});
    require(matches.size() == 1 && documents[matches[0].index].id == 12, "a document matches through any of its texts");
    const usize asked = service.requests;
    std::ignore = co_await retriever.search("task", "无语", documents, {.top_k = 3});
    require(service.requests == asked, "a repeated query is not embedded again");

    service.down = true;
    retrieval::Retriever offline(offline_config, *db);
    offline.backend = service.backend();
    matches = co_await offline.search("task", "周六打MC", documents, {.top_k = 2, .min_similarity = 0.5, .lexical = true});
    require(matches.size() == 1 && documents[matches[0].index].id == 11, "the service down, keywords still find it");
    service.down = false;

    const auto alike = co_await retriever.similarities("芒果过敏", {"芒果冰淇淋", "打 MC"});
    require(alike && (*alike)[0] > 0.9 && (*alike)[1] < 0.2, "document to document likeness");
}

/// A wider pool goes to the reranker, whose order wins; its failure leaves the fused order.
Task<> test_rerank() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    FakeService service;
    const MemoryConfig config{.embedding_model = "fake"};
    retrieval::Retriever retriever(config, *db);
    retriever.backend = service.backend();
    require(!retriever.can_rerank(), "no reranker without rerank_url");

    bool down = false;
    std::vector<retrieval::RerankInput> seen;
    std::string seen_task;
    retriever.reranker = [&](std::string query, std::vector<retrieval::RerankInput> documents,
                             std::string task) -> Task<Result<std::vector<double>>> {
        if (down) co_return lighter::outcome_error(Error::protocol("down"));
        seen = documents;
        seen_task = std::move(task);
        // Prefers the document that also has a picture.
        std::vector<double> scores;
        for (const auto &document : documents) scores.push_back(document.image.empty() ? 0.1 : 0.9);
        co_return scores;
    };
    const std::vector<retrieval::Document> documents{
        {.id = 1, .texts = {"芒果过敏"}},
        {.id = 2, .texts = {"芒果冰淇淋", retrieval::image_input("/s/2.png")}},
        {.id = 3, .texts = {"打 MC"}},
    };
    auto matches = co_await retriever.search("task", "芒果", documents, {.top_k = 1, .rerank = 3, .rerank_task = "judge"});
    require(seen.size() == 3 && seen_task == "judge", "the reranker sees the wider pool and the task");
    require(seen[1].text == "芒果冰淇淋" && seen[1].image == "/s/2.png", "text and picture of a document");
    require(matches.size() == 1 && documents[matches[0].index].id == 2 && matches[0].relevance == 0.9, "the reranker's best is kept");

    down = true;
    matches = co_await retriever.search("task", "芒果", documents, {.top_k = 2, .rerank = 3});
    require(matches.size() == 2 && !matches[0].relevance && documents[matches[0].index].id != 3, "down, the fused order stands");
}

/// "field" mode sends the task beside the query; pictures travel as tagged paths.
Task<> test_instruction_field_and_images() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    FakeService service;
    const MemoryConfig config{.embedding_model = "fake", .instruction_mode = "field"};
    retrieval::Retriever retriever(config, *db);
    retriever.backend = service.backend();
    std::ignore = co_await retriever.query("Find stickers", "无语");
    require(service.last_batch == std::vector<std::string>{"无语"} && service.last_instruction == "Find stickers",
            "the query goes bare, the task as the instruction");
    std::ignore = co_await retriever.query("Find chats", "无语");
    require(service.requests == 2, "the same text under another task is another query");

    const auto picture = retrieval::image_input("/data/stickers/abc.png");
    require(picture.starts_with(retrieval::k_image_tag) && picture.ends_with("/data/stickers/abc.png"), "a tagged path");
    const auto found = co_await retriever.vectors({picture});
    require(found[0] && service.last_batch.front() == picture && service.last_instruction.empty(), "documents carry no instruction");
}

} // namespace

int main() {
    try {
        test_vectors_and_text();
        test_lexical();
        test_fusion();
        for (auto *test : {&test_cache, &test_search, &test_rerank, &test_instruction_field_and_images}) {
            lighter::EventLoop loop;
            auto task = test();
            loop.schedule(task);
            loop.run();
            std::ignore = task.value();
        }
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
