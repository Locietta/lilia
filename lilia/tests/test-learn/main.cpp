#include <cstdio>
#include <format>
#include <random>
#include <set>
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

    const auto filtered = learn::parse_learned({tool_call("record_expressions", R"({"items": [
            {"situation": "吐槽某个人很差劲", "style": "用「杂鱼」", "source": "1"},
            {"situation": "调侃对方菜或者不行", "style": "用\"你怎么这么区\"", "source": "1"},
            {"situation": "评价某人水平一般", "style": "用\"很菜\"", "source": "1"},
            {"situation": "看到离谱的事", "style": "说「绷」", "source": "1"}
        ]})")},
                                               3);
    require(filtered.size() == 1 && filtered[0].style == "用\"绷\"", "put-downs are dropped; quotes and verbs are normalized");
    require(learn::normalize_style("回“洗澡”") == "用\"洗澡\"", "回 and curly quotes");
    require(learn::normalize_style("句尾加\"捏\"") == "句尾加\"捏\"", "other forms stay");
    require(!learn::blocked_expression("表示受不了", "用\"我不行了\""), "ordinary phrases pass");
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
    const auto habits = chat::sample_habits(*store, 5, rng);
    require(habits.size() == 5 && std::set<std::string>(habits.begin(), habits.end()).size() == 5, "sampled without replacement");
    require(chat::sample_habits(*store, 100, rng).size() == 21, "never more than the pool");
    usize top = 0;
    for (int i = 0; i < 400; ++i)
        top += std::ranges::contains(chat::sample_habits(*store, 1, rng), std::string("当觉得很好笑时，可以用\"笑死\""));
    // Weight 5 against 20 others of weight 1: about 1/5 of single picks.
    require(top > 40 && top < 140, std::format("the most counted pattern is weighted up ({} of 400)", top));
    require(chat::sample_voice(*store, 20) == std::vector<std::string>({"笑死"}),
            "voice samples are the bare quoted phrases, without duplicates or unquoted styles");
    require(chat::quoted_phrase("用\"何意味\"") == "何意味" && chat::quoted_phrase("句尾加个括号").empty(), "quoted phrase");

    require(store->prune_expressions(5).has_value(), "prune");
    require(store->expressions(100)->size() == 5, "pruned to the most counted");
}

void test_review_and_selection() {
    const auto review = learn::build_review_request("REVIEW", {{"觉得离谱", "用\"笑死\""}, {"不想理人", "用\"别理我\""}});
    require(text_of(review.history[0]) == "REVIEW" &&
                text_of(review.history[1]) == "1. 当觉得离谱时，可以用\"笑死\"\n2. 当不想理人时，可以用\"别理我\"\n",
            "review lists the patterns numbered from 1");
    require(review.tools.size() == 1 && review.tools[0].name == "review_expressions", "review tool");
    require(learn::parse_picks({tool_call("review_expressions", R"({"reject": ["2", 1, "#2", "9", "x"]})")}, "review_expressions", "reject",
                               2, 2) == std::vector<usize>({1, 0}),
            "picks parse numbers, strings and #n, dedupe, drop out-of-range");
    require(learn::parse_picks({}, "review_expressions", "reject", 2, 2).empty(), "no tool call picks nothing");

    const auto selection = learn::build_selection_request("SELECT", {.recent = {"a: 抽卡又歪了", "莉莉娅（她自己）: 惨"},
                                                                     .target = "a: 抽卡又歪了",
                                                                     .intent = "吐槽运气",
                                                                     .candidates = {"当倒霉时，可以用\"我服了\""}});
    const auto user = text_of(selection.history[1]);
    require(user.find("a: 抽卡又歪了\n莉莉娅（她自己）: 惨") != std::string::npos &&
                user.find("最新的一条：a: 抽卡又歪了") != std::string::npos && user.find("她想表达的意思：吐槽运气") != std::string::npos &&
                user.find("1. 当倒霉时，可以用\"我服了\"") != std::string::npos,
            "the selector sees the chat, the target, the intent and numbered candidates");
    require(learn::parse_picks({tool_call("select_expressions", R"({"ids": ["1", "1"]})")}, "select_expressions", "ids", 1, 3) ==
                std::vector<usize>({0}),
            "selection picks");

    const auto history = chat::build_writer_history({
        .persona = "P",
        .rules = "R",
        .info = {.chat = {.type = "group", .id = 1}, .name = "莉莉娅"},
        .recent = {{.message_id = 5, .sender_name = "a", .text = "笑死"}},
        .target = {.message_id = 5, .sender_name = "a", .text = "笑死"},
        .plan = {.reply = true, .target = 5},
        .habits = {"当附和时，可以用\"确实\""},
    });
    require(text_of(history.back()).find("【表达习惯参考，请视情况自然地使用】\n- 当附和时，可以用\"确实\"") != std::string::npos,
            "picked habits reach the writer");
}

} // namespace

int main() {
    try {
        test_request();
        test_parse();
        test_learnable_text();
        test_store_and_sampling();
        test_review_and_selection();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
