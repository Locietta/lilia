#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <glaze/json.hpp>

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

void test_facts() {
    const std::vector<store::StoredMessage> own{message(1, 10, "阿紫", "我对芒果过敏 上次吃了全身起疹子"), message(2, 20, "阿紫", "芒果")};
    const std::vector<store::StoredMessage> context{message(3, 15, "老王", "你对什么过敏来着"), message(4, 25, "", "记住了", true)};
    const auto request = memory::build_fact_request("PROMPT", "阿紫", own, context, "莉莉娅");
    std::string user;
    for (const auto &part : request.history.back().parts) {
        if (const auto *text = std::get_if<provider::TextPart>(&part)) user += text->text;
    }
    require(user.find("目标人物（TA）：阿紫") != std::string::npos && user.find("1. 我对芒果过敏") != std::string::npos &&
                user.find("2. 芒果") != std::string::npos && user.find("老王: 你对什么过敏来着") != std::string::npos &&
                user.find("莉莉娅: 记住了") != std::string::npos,
            "own messages are numbered, the nearby chat is context");
    require(request.tools.size() == 1 && request.tools[0].name == "record_facts", "record_facts tool");

    provider::ToolCallItem call{.call = {.id = "t", .name = "record_facts"}};
    require(!glz::read_json(call.call.input, R"({"facts": [
        {"fact": "TA 对芒果过敏。", "source": "1", "quote": "我对芒果 过敏"},
        {"fact": "TA住在杭州", "source": "1", "quote": "我住杭州"},
        {"fact": "TA猫", "source": "1", "quote": "芒果"},
        {"fact": "TA喜欢吃芒果冰淇淋", "source": "#3", "quote": "芒果"},
        {"fact": "TA对芒果过敏很严重", "source": 2, "quote": "芒果"}
    ]})"),
            "fake tool input parses");
    const auto facts = memory::parse_facts({provider::OutputItem(call)}, own);
    require(facts.size() == 2 && facts[0].fact == "TA对芒果过敏" && facts[0].source == 0 && facts[1].source == 1,
            "quotes must be in the cited message (spaces ignored); too short facts and unknown sources are dropped");

    const std::vector<store::PersonFact> stored{{.fact = "TA住在杭州"}, {.fact = "对芒果过敏"}};
    require(memory::render_facts(stored) == "住在杭州；对芒果过敏", "facts render without the TA subject");

    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    require(db->add_fact({.person_id = 7,
                          .fact = "TA对芒果过敏",
                          .quote = "芒果",
                          .chat = {.type = "group", .id = 1},
                          .time = 10,
                          .model = "m",
                          .vector = "abcd"})
                .has_value(),
            "add fact");
    require(db->add_fact({.person_id = 7,
                          .fact = "TA住在杭州",
                          .quote = "杭州",
                          .chat = {.type = "group", .id = 1},
                          .time = 20,
                          .model = "m",
                          .vector = "efgh"})
                .has_value(),
            "add fact");
    auto loaded = db->facts(7, 1);
    require(loaded.has_value() && loaded->size() == 1 && loaded->front().fact == "TA住在杭州" && loaded->front().vector == "efgh",
            "newest fact first, limited, blob intact");
    require(db->facts(8, 0)->empty() && db->facts(7, 0)->size() == 2, "per person; 0 lists all");
}

void test_summaries() {
    const std::vector<store::ChatSummary> previous{{.summary = "阿紫说周六去成都"}, {.summary = ""}};
    const auto request = memory::build_summary_request("PROMPT", previous, "[10-05 21:00] a: 定了\n");
    std::string user;
    for (const auto &part : request.history.back().parts) {
        if (const auto *text = std::get_if<provider::TextPart>(&part)) user += text->text;
    }
    require(user.find("以前的摘要（只用来理解上下文，别照抄）：\n- 阿紫说周六去成都\n") != std::string::npos &&
                user.find("新的群聊记录：\n[10-05 21:00] a: 定了") != std::string::npos,
            "previous non-empty summaries come first, then the new lines");
    require(request.tools.size() == 1 && request.tools[0].name == "record_summary", "record_summary tool");

    provider::ToolCallItem call{.call = {.id = "t", .name = "record_summary"}};
    require(!glz::read_json(call.call.input, R"({"summary": "定了周六去成都", "recall_cues": ["聊到成都", "", "周六的安排"]})"), "parse");
    auto parsed = memory::parse_summary({provider::OutputItem(call)});
    require(parsed && parsed->summary == "定了周六去成都" && parsed->cues == std::vector<std::string>({"聊到成都", "周六的安排"}),
            "summary and non-empty cues");
    require(!glz::read_json(call.call.input, R"({"summary": "", "recall_cues": ["x"]})"), "parse");
    parsed = memory::parse_summary({provider::OutputItem(call)});
    require(parsed && parsed->summary.empty() && parsed->cues.empty(), "nothing new is a valid answer, without cues");
    require(!memory::parse_summary({}), "no tool call is no answer");

    const auto chunk = memory::summary_chunk(
        {.id = 3, .start_time = 1'791'032'700, .end_time = 1'791'036'300, .summary = "定了周六去成都", .vector = "v"}, 480);
    require(chunk.text == "[10-03 21:05 至 10-03 22:05 的聊天摘要] 定了周六去成都\n" && chunk.id == -3 && chunk.vector == "v",
            "a summary ranks as a chunk");
    const auto cues = memory::encode_vector({1, 2}) + memory::encode_vector({3, 4});
    require(memory::split_vectors(cues, 2) == std::vector<std::vector<float>>({{1, 2}, {3, 4}}), "cue vectors split by dimension");

    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    const store::ChatKey chat{.type = "group", .id = 1};
    for (const i64 last : {i64{10}, i64{20}, i64{30}}) {
        require(db->add_summary({.chat = chat, .first_row = last - 9, .last_row = last, .summary = std::to_string(last), .model = "m"})
                    .has_value(),
                "add summary");
    }
    require(*db->last_summary_row(chat) == 30, "last summarized row");
    auto newest = db->summaries(chat, 2);
    require(newest.has_value() && newest->size() == 2 && newest->front().summary == "20" && newest->back().summary == "30",
            "the newest summaries, oldest first");
    require(db->summaries(chat, 0)->size() == 3, "0 lists all");
}

} // namespace

int main() {
    try {
        test_vectors();
        test_keywords();
        test_chunks();
        test_rank_and_render();
        test_store();
        test_facts();
        test_summaries();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
