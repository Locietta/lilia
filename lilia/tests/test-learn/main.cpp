#include <cstdio>
#include <random>
#include <string>
#include <variant>
#include <vector>

#include <glaze/json.hpp>

#include <lilia/chat/context.h>
#include <lilia/chat/responder.h>
#include <lilia/learn/learner.h>
#include <lilia/onebot/message.h>
#include <lilia/store/store.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;

std::string text_of(const provider::Item &item) {
    std::string out;
    for (const auto &part : item.parts) {
        if (const auto *text = std::get_if<provider::TextPart>(&part)) out += text->text;
    }
    return out;
}

provider::OutputItem tool_call(std::string name, std::string json) {
    provider::ToolCallItem item{.call = {.id = "toolu_1", .name = std::move(name)}};
    require(!glz::read_json(item.call.input, json), "test tool input must parse");
    return item;
}

void test_request() {
    const auto request = learn::build_learner_request(
        "PROMPT", {{.sender = 7, .text = "笑死"}, {.sender = 9, .text = "急了急了"}, {.sender = 7, .text = "确实"}});
    require(text_of(request.history[0]) == "PROMPT", "prompt is the system prompt");
    require(text_of(request.history[1]) == "1. 群友1: 笑死\n2. 群友2: 急了急了\n3. 群友1: 确实\n",
            "speakers are anonymised and lines numbered");
    require(request.tools.size() == 1 && request.tools[0].name == "record_expressions", "one tool");
    const auto &items = request.tools[0].input_schema.properties.at("items");
    require(items.type == "array" && items.items && items.items->is_object(), "items carry an element schema");
    const auto schema = glz::write_json(request.tools[0].input_schema).value_or("");
    require(schema.find(R"("items":{"type":"array")") != std::string::npos && schema.find(R"("source")") != std::string::npos,
            "the element schema is serialized: " + schema);
}

void test_parse() {
    const auto learned = learn::parse_learned({tool_call("record_expressions", R"({"items": [
            {"situation": "觉得离谱的时候", "style": "用\"笑死 这也行\"", "source": "1"},
            {"situation": "调侃别人破防", "style": "用\"急了急了\"", "source": 2},
            {"situation": "没有来源", "style": "用\"嗯嗯\"", "source": "9"},
            {"situation": "", "style": "空场景", "source": "1"},
            {"situation": "太长", "style": "这是一段明显超过三十个字的总结内容所以应该被丢弃掉才对的那种很长很长的说法", "source": "1"}
        ]})")},
                                              3);
    require(learned.size() == 2, "invalid items are dropped");
    require(learned[0].situation == "觉得离谱" && learned[0].style == "用\"笑死 这也行\"", "first item; a trailing 的时候 is dropped");
    require(learned[1].style == "用\"急了急了\"", "numeric source accepted");
    require(learn::parse_learned({tool_call("record_expressions", R"({"items": []})")}, 3).empty(), "an empty list is fine");
    require(learn::parse_learned({}, 3).empty(), "no tool call learns nothing");
}

void test_learnable_text() {
    using onebot::text_segment;
    require(learn::learnable_text({onebot::at_segment(1), text_segment("  好耶 ")}) == "好耶", "text only, trimmed");
    require(learn::learnable_text({onebot::Segment{.type = "image"}}).empty(), "media-only messages are skipped");
    require(learn::learnable_text({text_segment("/lilia status")}).empty(), "commands are skipped");
    require(learn::learnable_text({text_segment(std::string(400, 'a'))}).empty(), "copypasta-length lines are skipped");
}

void test_store_and_sampling() {
    auto store = store::Store::open_memory();
    require(store.has_value(), "open");
    require(store->learn_expression("觉得离谱", "用\"笑死\"", 10).has_value(), "learn");
    require(store->learn_expression("觉得很好笑", "用\"笑死\"", 20).has_value(), "learn again");
    for (int i = 0; i < 20; ++i)
        require(store->learn_expression("场景" + std::to_string(i), "说法" + std::to_string(i), 30).has_value(), "learn");
    auto rows = store->expressions(100);
    require(rows.has_value() && rows->size() == 21, "one row per style");
    require((*rows)[0].style == "用\"笑死\"" && (*rows)[0].count == 2 && (*rows)[0].situation == "觉得很好笑",
            "a repeated style counts up and keeps the newest situation");
    require(store->expressions(0)->size() == 21, "limit 0 lists everything");
    auto random = store->random_expressions(1, 0);
    require(random.has_value() && random->size() == 20 &&
                std::ranges::none_of(*random, [](const store::Expression &e) { return e.style == "用\"笑死\""; }),
            "random picks skip the top patterns");

    std::mt19937 rng(3);
    const auto habits = chat::sample_habits(*store, 3, rng);
    require(habits.size() == 3, "sample size");
    require(std::ranges::contains(habits, std::string("当觉得很好笑时，可以用\"笑死\"")), "the most counted pattern is always offered");
    require(chat::sample_habits(*store, 2, rng).size() == 2, "fewer than three candidates are all random");

    require(store->prune_expressions(5).has_value(), "prune");
    require(store->expressions(100)->size() == 5, "pruned to the most counted");
}

void test_planner_habits() {
    const auto request = chat::build_planner_request({
        .prompt = "P",
        .info = {.chat = {.type = "group", .id = 1}, .name = "莉莉娅"},
        .recent = {{.chat = {.type = "group", .id = 1}, .message_id = 5, .sender_id = 2, .sender_name = "a", .text = "笑死"}},
        .fresh = {5},
        .habits = {"当觉得离谱时，可以用\"笑死\"", "当附和时，可以用\"确实\""},
    });
    const auto user = text_of(request.history[1]);
    require(user.find("E1 当觉得离谱时") != std::string::npos && user.find("E2 当附和时") != std::string::npos, "habits are listed");
    const auto plan = chat::parse_plan(
        {tool_call("reply", R"({"target": "#1", "style": "short", "intent": "x", "habits": ["E2", "e9", 1, "E2", "E1"]})")}, request.ids);
    require(plan.reply && plan.habits == std::vector<usize>({1, 8}), "picks parse, dedupe and stop at two");

    const auto history = chat::build_writer_history({
        .persona = "P",
        .rules = "R",
        .info = {.chat = {.type = "group", .id = 1}, .name = "莉莉娅"},
        .recent = {{.message_id = 5, .sender_name = "a", .text = "笑死"}},
        .target = {.message_id = 5, .sender_name = "a", .text = "笑死"},
        .plan = {.reply = true, .target = 5},
        .habits = {"当附和时，可以用\"确实\""},
    });
    require(text_of(history.back()).find("可以参考的说法") != std::string::npos, "picked habits reach the writer");
}

} // namespace

int main() {
    try {
        test_request();
        test_parse();
        test_learnable_text();
        test_store_and_sampling();
        test_planner_habits();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
