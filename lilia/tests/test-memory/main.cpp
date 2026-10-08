#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <lilia/memory/memory.h>
#include <lilia/store/store.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;

store::StoredMessage message(i64 row, i64 time, std::string name, std::string text, bool self = false) {
    return {.chat = {.type = "group", .id = 1},
            .message_id = row,
            .sender_id = 100 + row,
            .sender_name = std::move(name),
            .time = time,
            .text = std::move(text),
            .from_self = self,
            .row_id = row};
}

void test_vectors() {
    const std::vector<float> v{0.5f, -1.25f, 3.0f};
    require(memory::decode_vector(memory::encode_vector(v)) == v, "vectors round-trip through bytes");
    require(memory::encode_vector(v).size() == 12, "float32 bytes");
    require(std::abs(memory::cosine({1, 0}, {1, 0}) - 1) < 1e-9 && std::abs(memory::cosine({1, 0}, {0, 1})) < 1e-9, "cosine");
    require(memory::cosine({1, 0}, {1, 0, 0}) == 0 && memory::cosine({}, {}) == 0, "mismatched or empty vectors score 0");
}

void test_keywords() {
    require(memory::keywords("谁说过 MC 整合包？") == std::vector<std::string>({"谁说过", "MC", "整合包"}), "CJK runs and ASCII words");
    require(memory::keywords("a 我 Godot4!") == std::vector<std::string>({"Godot4"}), "single characters are dropped");
}

void test_chunks() {
    std::vector<store::StoredMessage> messages;
    for (i64 i = 1; i <= 8; ++i) messages.push_back(message(i, 1000 + i * 10, "a", "x"));
    messages.push_back(message(9, 1000 + 80 + 3600, "b", "隔了一个小时"));
    auto chunks = memory::complete_chunks(messages, 6, 1800, 1000 + 3700);
    require(chunks.size() == 2 && chunks[0].size() == 6 && chunks[1].size() == 2, "full chunks, and a chunk ends at a long pause");
    require(chunks[1].back().row_id == 8, "the growing tail after the pause waits");
    chunks = memory::complete_chunks(messages, 6, 1800, 1000 + 80 + 3600 + 1800);
    require(chunks.size() == 3 && chunks[2].front().row_id == 9, "a tail the chat has moved on from is complete");
}

void test_rank_and_render() {
    std::vector<store::MemoryChunk> chunks{
        {.id = 1, .text = "[10-05 21:03] a: 我对芒果过敏\n", .vector = memory::encode_vector({1, 0})},
        {.id = 2, .text = "[10-06 20:00] b: 周六一起打 MC\n", .vector = memory::encode_vector({0.97f, 0.2431f})},
        {.id = 3, .text = "[10-07 10:00] c: 早\n", .vector = memory::encode_vector({0, 1})},
    };
    auto hits = memory::rank(chunks, {1, 0}, {}, 2, 0.5);
    require(hits.size() == 2 && hits[0].chunk.id == 1 && hits[1].chunk.id == 2, "best first, limited, below min_score dropped");
    hits = memory::rank(chunks, {1, 0}, {"MC"}, 1, 0);
    require(hits.size() == 1 && hits[0].chunk.id == 2 && std::abs(hits[0].score - 1.02) < 1e-3,
            "a literal keyword hit decides between close candidates");
    require(memory::render_hits({}) == "没找到相关的聊天记录。", "no hits");
    const auto text = memory::render_hits(memory::rank(chunks, {1, 0}, {}, 2, 0.5));
    require(text.find("我对芒果过敏") < text.find("---") && text.find("---") < text.find("周六一起打 MC"), "hits are separated");

    const std::vector<store::StoredMessage> lines{message(1, 1'791'032'700, "a", "在吗"), message(2, 1'791'032'760, "", "在", true)};
    require(memory::display_text(lines, "莉莉娅", 480) == "[10-03 21:05] a: 在吗\n[10-03 21:06] 莉莉娅: 在\n",
            "displayed form with local time");
    require(memory::embed_text(lines, "莉莉娅") == "a: 在吗\n莉莉娅: 在\n", "embedded form without times");
}

void test_store() {
    auto store = store::Store::open_memory();
    require(store.has_value(), "open");
    const store::ChatKey chat{.type = "group", .id = 1};
    for (i64 i = 1; i <= 3; ++i) {
        require(store->add_message({.chat = chat, .message_id = i, .sender_id = 7, .sender_name = "a", .time = i, .text = "t"}).has_value(),
                "add message");
    }
    auto after = store->messages_after(chat, 1, 10);
    require(after.has_value() && after->size() == 2 && after->front().row_id == 2, "messages after a row, with row ids");
    require(store->add_chunk({.chat = chat, .first_row = 1, .last_row = 2, .text = "x", .model = "m1", .vector = "abcd"}).has_value(),
            "add chunk");
    require(store->add_chunk({.chat = chat, .first_row = 1, .last_row = 3, .text = "y", .model = "m2", .vector = "efgh"}).has_value(),
            "add chunk");
    require(*store->last_chunk_row(chat, "m1") == 2 && *store->last_chunk_row(chat, "m3") == 0, "last covered row per model");
    auto chunks = store->chunks(chat, "m2");
    require(chunks.has_value() && chunks->size() == 1 && chunks->front().vector == "efgh", "chunks per model, blob intact");
    require(store->delete_chunks_except("m2").has_value() && store->chunks(chat, "m1")->empty() && store->chunks(chat, "m2")->size() == 1,
            "other models' chunks are dropped");
}

} // namespace

int main() {
    try {
        test_vectors();
        test_keywords();
        test_chunks();
        test_rank_and_render();
        test_store();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
