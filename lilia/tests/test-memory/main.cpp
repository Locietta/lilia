#include <cstdio>
#include <string>
#include <vector>

#include <glaze/json.hpp>

#include <lighter/async/io/loop.h>

#include <lilia/memory/memory.h>
#include <lilia/store/store.h>
#include <lilia/tests/support/fake_provider.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;
using lighter::Task;

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

void test_render() {
    const std::vector<memory::Hit> hits{{.chunk = {.id = 1, .text = "[10-05 21:03] a: 我对芒果过敏\n"}, .score = 0.8},
                                        {.chunk = {.id = 2, .text = "[10-06 20:00] b: 周六一起打 MC\n"}, .score = 0.6}};
    require(memory::render_hits({}) == "没找到相关的聊天记录。", "no hits");
    const auto text = memory::render_hits(hits);
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
    require(*store->last_chunk_row(chat) == 0, "nothing covered yet");
    require(store->add_chunk({.chat = chat, .first_row = 1, .last_row = 2, .text = "[..] a: x", .search_text = "a: x"}).has_value(),
            "add chunk");
    require(*store->last_chunk_row(chat) == 2 && *store->last_chunk_row({.type = "group", .id = 2}) == 0, "last covered row per chat");
    auto chunks = store->chunks(chat);
    require(chunks.has_value() && chunks->size() == 1 && chunks->front().search_text == "a: x" && chunks->front().text == "[..] a: x",
            "chunks keep both forms");
    require(*store->search_texts() == std::vector<std::string>{"a: x"}, "every chunk's search text");
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
    require(
        db->add_fact({.person_id = 7, .fact = "TA对芒果过敏", .quote = "芒果", .chat = {.type = "group", .id = 1}, .time = 10}).has_value(),
        "add fact");
    require(
        db->add_fact({.person_id = 7, .fact = "TA住在杭州", .quote = "杭州", .chat = {.type = "group", .id = 1}, .time = 20}).has_value(),
        "add fact");
    auto loaded = db->facts(7, 1);
    require(loaded.has_value() && loaded->size() == 1 && loaded->front().fact == "TA住在杭州" && loaded->front().person_id == 7,
            "newest fact first, limited");
    require(db->facts(8, 0)->empty() && db->facts(7, 0)->size() == 2 && db->all_facts()->size() == 2, "per person; 0 lists all");
}

void test_summaries() {
    const std::vector<store::ChatSummary> previous{{.summary = "阿紫说周六去成都"}, {.summary = ""}};
    const auto request = memory::build_summary_request("PROMPT", previous, "[10-05 21:00] a: 定了\n");
    std::string user;
    for (const auto &part : request.history.back().parts) {
        if (const auto *text = std::get_if<provider::TextPart>(&part)) user += text->text;
    }
    require(user.find("以前的摘要（只用来理解上下文，别照抄）：\n- 阿紫说周六去成都\n") != std::string::npos &&
                user.find("新的聊天记录：\n[10-05 21:00] a: 定了") != std::string::npos,
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

    const auto chunk =
        memory::summary_chunk({.id = 3, .start_time = 1'791'032'700, .end_time = 1'791'036'300, .summary = "定了周六去成都"}, 480);
    require(chunk.text == "[10-03 21:05 至 10-03 22:05 的聊天摘要] 定了周六去成都\n" && chunk.id == -3, "a summary ranks as a chunk");
    require(memory::summary_texts({.summary = "定了周六去成都", .cues = "聊到成都\n\n周六的安排"}) ==
                    std::vector<std::string>({"定了周六去成都", "聊到成都", "周六的安排"}) &&
                memory::summary_texts({.summary = "", .cues = "x"}).empty(),
            "a summary is found by itself and its cues");

    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    const store::ChatKey chat{.type = "group", .id = 1};
    for (const i64 last : {i64{10}, i64{20}, i64{30}}) {
        require(db->add_summary({.chat = chat, .first_row = last - 9, .last_row = last, .summary = std::to_string(last)}).has_value(),
                "add summary");
    }
    require(*db->last_summary_row(chat) == 30, "last summarized row");
    auto newest = db->summaries(chat, 2);
    require(newest.has_value() && newest->size() == 2 && newest->front().summary == "20" && newest->back().summary == "30",
            "the newest summaries, oldest first");
    require(db->summaries(chat, 0)->size() == 3 && db->all_summaries()->size() == 3, "0 lists all");
    require(db->add_summary({.chat = chat, .first_row = 31, .last_row = 40, .summary = "w", .kind = std::string(store::k_window_summary)})
                .has_value(),
            "add a window summary");
    require(*db->last_summary_row(chat) == 30 && *db->last_summary_row(chat, store::k_window_summary) == 40 &&
                db->summaries(chat, 0)->size() == 3 && db->summaries(chat, 0, store::k_window_summary)->front().summary == "w" &&
                db->all_summaries()->size() == 4,
            "each kind keeps its own rows");
}

/// Memory on a fake embedding service: a direction per topic word.
retrieval::Retriever::Backend fake_embeddings() {
    return [](std::vector<std::string> texts, std::string) -> Task<Result<std::vector<retrieval::Vector>>> {
        static const std::vector<std::pair<std::string, retrieval::Vector>> k_topics{
            {"芒果", {1, 0, 0, 0}}, {"成都", {0, 1, 0, 0}}, {"MC", {0, 0, 1, 0}}};
        std::vector<retrieval::Vector> out;
        for (const auto &text : texts) {
            retrieval::Vector vector{0, 0, 0, 0.1f};
            for (const auto &[word, direction] : k_topics) {
                if (text.find(word) == std::string::npos) continue;
                for (usize i = 0; i < 4; ++i) vector[i] += direction[i];
            }
            out.push_back(std::move(vector));
        }
        co_return out;
    };
}

Task<> test_memory() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    const MemoryConfig config{.chunk_messages = 2, .search_results = 2, .min_score = 0.5, .recall_messages = 2, .recall_score = 0.6};
    retrieval::Retriever retriever(config, *db);
    retriever.backend = fake_embeddings();
    memory::Memory memory(config, "莉莉娅", 480, *db, retriever);
    memory.now = [] { return i64{1'000'000}; };
    const store::ChatKey chat{.type = "group", .id = 1};
    const std::vector<std::pair<std::string, std::string>> lines{{"阿紫", "我对芒果过敏"}, {"老王", "记住了"}, {"阿紫", "周六一起打 MC"},
                                                                 {"老王", "好"},           {"阿紫", "早"},     {"老王", "早"}};
    for (usize i = 0; i < lines.size(); ++i) {
        require(db->add_message({.chat = chat,
                                 .message_id = static_cast<i64>(i + 1),
                                 .sender_id = 7,
                                 .sender_name = lines[i].first,
                                 .time = static_cast<i64>(1000 + i),
                                 .text = lines[i].second})
                    .has_value(),
                "add message");
    }
    co_await memory.index(chat);
    require(db->chunks(chat)->size() == 3 && db->search_texts()->front() == "阿紫: 我对芒果过敏\n老王: 记住了\n", "chunked and indexed");
    require(memory::embedded_texts(*db).size() == 3, "what memory has embedded");

    auto hits = co_await memory.search(chat, "谁对芒果过敏");
    require(hits.size() == 1 && hits[0].chunk.text.contains("我对芒果过敏") && hits[0].score > 0.9, "found by meaning, gated");
    hits = co_await memory.search(chat, "MC");
    require(hits.size() == 1 && hits[0].chunk.text.contains("打 MC"), "another topic");

    // Rolling summaries compete in search.
    require(db->add_summary({.chat = chat,
                             .first_row = 1,
                             .last_row = 2,
                             .end_time = 1001,
                             .summary = "阿紫说周末去成都",
                             .cues = "聊到成都\n成都的安排"})
                .has_value(),
            "add summary");
    hits = co_await memory.search(chat, "成都");
    require(hits.size() == 1 && hits[0].chunk.id < 0 && hits[0].chunk.text.contains("聊天摘要"), "a summary is a hit too");

    // Window summaries: what the planner's window dropped.
    auto summarizer = std::make_shared<test::FakeProvider::State>();
    summarizer->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        std::string seen;
        for (const auto &part : history.back().parts) {
            if (const auto *text = std::get_if<provider::TextPart>(&part)) seen += text->text;
        }
        const bool chengdu = seen.contains("成都");
        return std::vector{test::tool_call("record_summary", chengdu ? R"({"summary":"阿紫说周末去成都","recall_cues":["聊到成都"]})" :
                                                                       R"({"summary":"阿紫和老王在闲聊","recall_cues":["闲聊"]})")};
    };
    memory.summarizer = test::fake_provider(summarizer);
    require(db->add_message({.chat = chat, .message_id = 7, .sender_id = 7, .sender_name = "阿紫", .time = 1006, .text = "周末去成都"})
                .has_value(),
            "add message");
    co_await memory.summarize_window(chat, 7, 4);
    auto windows = db->summaries(chat, 0, store::k_window_summary);
    require(windows->size() == 1 && windows->front().first_row == 4 && windows->front().last_row == 7,
            "the newest lines up to the anchor, at most the span");
    co_await memory.summarize_window(chat, 7, 4);
    require(db->summaries(chat, 0, store::k_window_summary)->size() == 1, "lines are summarised once");
    require(memory.earlier(chat).contains("阿紫说周末去成都"), "the newest lead the planner's log");
    std::vector<store::StoredMessage> recent{{.chat = chat, .sender_name = "老王", .text = "成都好玩吗", .row_id = 20},
                                             {.chat = chat, .sender_name = "阿紫", .text = "还行", .row_id = 21}};
    require((co_await memory.recall(chat, recent)).empty(), "one the planner already sees is not recalled");
    for (usize i = 0; i < memory::k_earlier_summaries; ++i) {
        const auto row = static_cast<i64>(8 + i);
        require(db->add_message({.chat = chat, .message_id = row, .sender_id = 7, .sender_name = "老王", .time = 1000 + row, .text = "早"})
                    .has_value(),
                "add message");
        co_await memory.summarize_window(chat, row, 4);
    }
    const auto shown = memory.earlier(chat);
    require(!shown.contains("成都") && shown.contains("闲聊"), "only the newest few are shown: " + shown);
    require((co_await memory.recall(chat, recent)).contains("阿紫说周末去成都"), "an older one is recalled by its cue");
    recent[0].text = "今天吃什么";
    require((co_await memory.recall(chat, recent)).empty(), "nothing alike, nothing recalled");

    // Facts: a rephrased duplicate is kept out, or replaces the old one.
    store::PersonFact fact{.person_id = 7, .fact = "TA对芒果过敏", .quote = "芒果", .chat = chat, .time = 10};
    require(co_await memory.remember(fact), "a new fact");
    fact.fact = "TA吃芒果会过敏";
    require(!co_await memory.remember(fact), "the same meaning again");
    fact.time = 20;
    const bool replaced = co_await memory.remember(fact, true);
    auto known = db->facts(7, 0);
    require(replaced && known->size() == 1 && known->front().fact == "TA吃芒果会过敏",
            "or it replaces the old one: " + std::to_string(known->size()));
    fact.fact = "TA周末去成都";
    const bool added = co_await memory.remember(fact);
    require(added && db->facts(7, 0)->size() == 2, "a different fact");
}

} // namespace

int main() {
    try {
        test_chunks();
        test_render();
        test_store();
        test_facts();
        test_summaries();
        lighter::EventLoop loop;
        auto task = test_memory();
        loop.schedule(task);
        loop.run();
        std::ignore = task.value();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
