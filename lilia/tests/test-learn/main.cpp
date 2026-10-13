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
#include <lilia/learn/frequency.h>
#include <lilia/learn/learner.h>
#include <lilia/learn/segment.h>
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

const std::vector<learn::Line> k_lines{
    {.sender = 7, .text = "笑死 这也行"},
    {.sender = 9, .text = "急了急了"},
    {.text = "我也觉得", .self = true},
    {.sender = 7, .text = "这卡太T0了 xswl"},
};
const std::vector<std::string> k_names{"莉莉娅", "lilia"};

void test_request() {
    const auto request = learn::build_learner_request("PROMPT", k_lines, "莉莉娅");
    require(text_of(request.history[0]) == "PROMPT", "prompt is the system prompt");
    require(text_of(request.history[1]) ==
                "1. 群友1: 笑死 这也行\n2. 群友2: 急了急了\n3. [SELF] 莉莉娅: 我也觉得\n4. 群友1: 这卡太T0了 xswl\n",
            "speakers are anonymised, her own lines marked [SELF], lines numbered");
    require(request.tools.size() == 1 && request.tools[0].name == "record_expressions", "one tool");
    const auto schema = glz::write_json(request.tools[0].input_schema).value_or("");
    require(schema.find(R"("items":{"type":"array")") != std::string::npos && schema.find(R"("source")") != std::string::npos,
            "the element schema is serialized: " + schema);
}

void test_parse() {
    const auto learned = learn::parse_learned({tool_call("record_expressions", R"({"items": [
            {"situation": "觉得离谱的时候", "style": "用\"笑死 这也行\"", "source": "1"},
            {"situation": "调侃别人破防", "style": "使用“急了急了”", "source": 2},
            {"situation": "学她自己", "style": "用\"我也觉得\"", "source": "3"},
            {"situation": "没有来源", "style": "用\"嗯嗯\"", "source": "9"},
            {"situation": "发图", "style": "用[图片]回应", "source": "1"},
            {"situation": "叫她", "style": "用\"莉莉娅\"", "source": "1"},
            {"situation": "", "style": "空场景", "source": "1"}
        ]})")},
                                              k_lines, k_names);
    require(learned && learned->size() == 2, "self, missing, media, her name and empty items are dropped");
    require((*learned)[0].situation == "觉得离谱" && (*learned)[0].source == 0, "a trailing 的时候 is dropped; the source is kept");
    require((*learned)[1].style == "用\"急了急了\"", "使用 and curly quotes are normalised");

    std::string many = R"({"items": [)";
    for (int i = 0; i < 21; ++i)
        many += std::format(R"({}{{"situation": "s{}", "style": "用\"x{}\"", "source": "1"}})", i ? "," : "", i, i);
    many += "]}";
    require(!learn::parse_learned({tool_call("record_expressions", many)}, k_lines, k_names), "more than 20 items discard the batch");
    require(learn::parse_learned({}, k_lines, k_names)->empty(), "no tool call learns nothing");

    const std::vector<learn::Line> spoken{
        {.sender = 7, .text = "这人纯纯废物"},
        {.sender = 9, .text = "你怎么这么区"},
        {.sender = 7, .text = "绷"},
    };
    const auto filtered = learn::parse_learned({tool_call("record_expressions", R"({"items": [
            {"situation": "吐槽某个人很差劲", "style": "用「废物」", "source": "1"},
            {"situation": "调侃对方菜或者不行", "style": "用\"你怎么这么区\"", "source": "2"},
            {"situation": "看到离谱的事", "style": "说「绷」", "source": "3"}
        ]})")},
                                               spoken, k_names);
    require(filtered && filtered->size() == 1 && filtered->front().style == "用\"绷\"", "put-downs are dropped");
    require(learn::normalize_style("回“洗澡”") == "用\"洗澡\"" && learn::normalize_style("句尾加\"捏\"") == "句尾加\"捏\"", "normalising");
    require(!learn::blocked_expression("开玩笑逗人", "用\"杂鱼~\"") && !learn::blocked_expression("评价新卡强度不够", "用\"这卡有点区\""),
            "playful 杂鱼 and 区 about a card pass");
    require(learn::blocked_expression("调侃对方不行", "用\"你怎么这么区\"") && learn::blocked_expression("觉得离谱", "用\"弱智吧\""),
            "aimed at a person or a slur is blocked");
    require(learn::blocked_expression("调侃对方", "用\"你真菜\"") && learn::blocked_expression("调侃对方", "用\"你也太区了吧\"") &&
                learn::blocked_expression("觉得离谱", "用\"你真sb\""),
            "jieba's filler words (真菜) and HMM runs (太区) still count; sb after CJK is a word");
    require(!learn::blocked_expression("问对方有什么不同", "用\"你看这个区别\"") &&
                !learn::blocked_expression("给对方指路", "用\"你看菜单\"") &&
                !learn::blocked_expression("教别人看网页", "用\"你先滚动到底\"") &&
                !learn::blocked_expression("聊旅行", "用\"你去过Lisbon吗\""),
            "区别, 菜单, 滚动 are other words; sb inside an English word is not a slur");
}

void test_grounding() {
    usize ungrounded = 0;
    const auto learned = learn::parse_learned({tool_call("record_expressions", R"({"items": [
            {"situation": "觉得离谱", "style": "用\"笑死……这也行\"", "source": "1"},
            {"situation": "夸卡强", "style": "用「太T0了」和「XSWL」", "source": "4"},
            {"situation": "觉得可以", "style": "用'这也行'", "source": "1"},
            {"situation": "觉得离谱", "style": "用\"笑死，这也行\"", "source": "1"},
            {"situation": "调侃别人", "style": "用\"急了急了\"", "source": "1"},
            {"situation": "夸卡强", "style": "用\"T\"", "source": "4"},
            {"situation": "觉得好笑", "style": "说笑死", "source": "1"},
            {"situation": "觉得好笑", "style": "用\"\"", "source": "1"}
        ]})")},
                                              k_lines, k_names, &ungrounded);
    require(learned && learned->size() == 3 && (*learned)[1].style == "用\"太T0了\"和\"XSWL\"" && (*learned)[2].style == "用'这也行'",
            "every quoted part, cut at ellipses, is in the source line, ignoring case");
    require(ungrounded == 5, std::format("rewordings, other lines, partial ASCII words and unquoted styles are dropped: {}", ungrounded));
}

void test_review() {
    auto rejected = learn::parse_review(
        {tool_call("review_expressions", R"({"reject": [{"id": "2", "reason": "圈内黑话"}, {"id": "9", "reason": "x"}]})")}, 2);
    require(rejected && rejected->size() == 1 && rejected->front().index == 1 && rejected->front().reason == "圈内黑话",
            "rejects with reasons; out-of-range ids are dropped");
    rejected = learn::parse_review({tool_call("review_expressions", R"({"reject": ["1"]})")}, 2);
    require(rejected && rejected->size() == 1 && rejected->front().index == 0, "a bare id list is tolerated");
    require(!learn::parse_review({}, 2), "no review is no answer");
}

void test_jargon() {
    const auto request = learn::build_jargon_request("JARGON", k_lines, "莉莉娅");
    require(request.tools.front().name == "record_jargon", "record_jargon tool");
    const auto found = learn::parse_jargon({tool_call("record_jargon", R"({"items": [
            {"content": "T0", "source": "4"},
            {"content": "xswl", "source": "4"},
            {"content": "x", "source": "4"},
            {"content": "群友1", "source": "1"},
            {"content": "不存在", "source": "2"},
            {"content": "T0", "source": "4"}
        ]})")},
                                           k_lines, k_names);
    require(found.size() == 2 && found[0].content == "T0" && found[1].content == "xswl",
            "short, real, not names, deduplicated; the term must occur in its line");
    const auto hits = learn::known_hits(k_lines, {"急了", "T0", "也觉得", "T", "xs"});
    require(hits.size() == 2 && hits[0].content == "急了" && hits[0].source == 1 && hits[1].content == "T0" && hits[1].source == 3,
            "known terms in others' lines; ASCII as whole words; her own lines skipped");
    const auto snippet = learn::evidence_snippet(k_lines, 3, "莉莉娅");
    require(snippet == "群友1: 笑死 这也行\n群友2: 急了急了\n[SELF] 莉莉娅: 我也觉得\n群友1: 这卡太T0了 xswl\n",
            "three lines on each side, anonymised");

    require(!learn::inference_due(3, 0) && learn::inference_due(4, 0) && !learn::inference_due(5, 4) && learn::inference_due(9, 4) &&
                learn::inference_due(30, 8) && !learn::inference_due(30, 25) && learn::inference_due(120, 25),
            "inference at 4, 8, 25 and 100 sightings");

    const store::Jargon term{.id = 1, .content = "T0", .meaning = "最强的一档"};
    const auto infer = learn::build_inference_request("{content}|{contexts}|{previous}|{previous_instruction}|{name}", term,
                                                      {{.snippet = "群友1: 这卡太T0了\n"}}, "莉莉娅");
    const auto infer_text = text_of(infer.history.back());
    require(infer_text.find("T0|【对话片段 1】\n群友1: 这卡太T0了") == 0 && infer_text.find("上一次推断的含义") != std::string::npos &&
                infer_text.find("|莉莉娅") != std::string::npos,
            "inference sees the evidence and the previous meaning");
    auto meaning = learn::parse_meaning({tool_call("record_meaning", R"({"meaning": "强度第一档", "no_info": false})")});
    require(meaning && meaning->meaning == "强度第一档" && !meaning->no_info, "meaning");
    meaning = learn::parse_meaning({tool_call("record_meaning", R"({"meaning": "", "no_info": false})")});
    require(meaning && meaning->no_info, "an empty meaning is no information");
    require(learn::parse_similar({tool_call("record_comparison", R"({"is_similar": true, "reason": "一样"})")}) == true &&
                !learn::parse_similar({}),
            "comparison");

    const std::vector<store::Jargon> terms{
        {.content = "T0", .meaning = "最强一档", .is_jargon = 1, .count = 5},
        {.content = "xswl", .meaning = "笑死我了\n来源：拼音首字母缩写，常用于……", .is_jargon = 1, .count = 9},
        {.content = "急了", .meaning = "", .is_jargon = 1, .count = 50},
        {.content = "也觉得", .meaning = "普通", .is_jargon = 0, .count = 50},
    };
    const std::vector<store::StoredMessage> log{{.text = "这卡太T0了 xswl"}, {.text = "也觉得", .from_self = true}};
    const auto notes = learn::jargon_reference(terms, log, {}, 10);
    require(notes.find("以下黑话来自当前上下文中其他用户消息的机械匹配") == 0 && notes.find("1. xswl：笑死我了\n") != std::string::npos &&
                notes.find("2. T0：最强一档\n") != std::string::npos && notes.find("急了") == std::string::npos &&
                notes.find("也觉得") == std::string::npos,
            "slang with a meaning that others used, most seen first, each meaning's first sentence only");
    require(learn::jargon_reference(terms, {}, {}, 10).empty(), "nothing matched");
    const auto boosted = learn::jargon_reference(terms, log, {{.term = "t0", .occurrence_count = 3, .rank = 1}}, 10);
    require(boosted.find("1. T0：最强一档，同时命中高频词\n") != std::string::npos &&
                boosted.find("2. xswl：笑死我了\n") != std::string::npos,
            "slang that is also a frequent term of the chat comes first");
    const std::vector<store::StoredMessage> later{{.text = "xswl"}, {.text = "T0 还是 xswl"}};
    const std::vector<store::Jargon> even{{.content = "T0", .meaning = "a", .is_jargon = 1, .count = 5},
                                          {.content = "xswl", .meaning = "b", .is_jargon = 1, .count = 5}};
    require(learn::jargon_reference(even, later, {}, 10).find("1. xswl：b\n2. T0：a\n") != std::string::npos,
            "equal counts: the term seen earlier first");
}

void test_sharing() {
    require(learn::shares({}, 1, 2) && learn::shares({{1, 2}}, 1, 2) && !learn::shares({{1, 2}}, 1, 3) && learn::shares({{0}}, 5, 6) &&
                learn::shares({{1}}, 1, 1) && learn::shares({{1}}, 3, 0),
            "sharing groups; 0 is everyone; legacy (0) patterns are everyone's");
    require(learn::expression_text("觉得离谱", "用\"笑死\"") == "情景：觉得离谱\n风格：用\"笑死\"", "what a pattern is embedded as");
}

void test_learnable_text() {
    using onebot::text_segment;
    require(learn::learnable_text({onebot::at_segment(1), text_segment("  好耶 ")}) == "好耶", "text only, trimmed");
    require(learn::learnable_text({onebot::Segment{.type = "image"}}).empty(), "media-only messages are skipped");
    require(learn::learnable_text({text_segment("/lilia status")}).empty(), "commands are skipped");
    require(learn::learnable_text({text_segment(std::string(400, 'a'))}).empty(), "copypasta-length lines are skipped");
}

void test_store() {
    auto store = store::Store::open_memory();
    require(store.has_value(), "open");
    const store::ChatKey a{.type = "group", .id = 1};
    const store::ChatKey b{.type = "group", .id = 2};
    auto first = store->learn_expression(a, "觉得离谱", "用\"笑死\"", 10);
    auto again = store->learn_expression(a, "觉得很好笑", "用\"笑死\"", 20);
    auto same = store->learn_expression(a, "觉得离谱", "用\"笑死\"", 30);
    auto other = store->learn_expression(b, "觉得离谱", "用\"笑死\"", 40);
    require(first.has_value() && again.has_value() && *first == *again && *again == *same && *other != *first, "merged per chat");
    auto rows = store->expressions(0);
    const auto &row = rows->front().chat.id == 1 ? rows->front() : rows->back();
    require(rows->size() == 2 && row.count == 3 && row.situation == "觉得离谱" && row.situations == "觉得离谱\n觉得很好笑",
            "counts up; every situation kept once; the newest leads");
    require(store->touch_expressions({*first}, 99).has_value() && store->set_expression_checked(*first, true).has_value(), "touch, check");
    rows = store->expressions(0);
    const auto checked = std::ranges::find(*rows, *first, &store::Expression::id);
    require(checked->last_active == 99 && checked->checked && checked->modified_by == "USER", "stored");

    require(store->add_expression_review({.time = 1, .chat = a, .situation = "s", .style = "y", .verdict = "rejected", .reason = "r"})
                .has_value(),
            "review log");
    auto reviews = store->expression_reviews("rejected", 10);
    require(reviews->size() == 1 && reviews->front().reason == "r" && (*store->expression_review(reviews->front().id))->style == "y",
            "rejected reviews");

    for (int i = 0; i < 12; ++i) {
        require(store->add_jargon_hit("T0", i % 2 ? a : b, std::format("片段{}", i), 100 + i, 10).has_value(), "jargon hit");
    }
    auto term = store->jargon_by_content("T0");
    require(term && *term && (*term)->count == 12, "sightings count");
    require(store->jargon_evidence((*term)->id, 0)->size() == 10 && store->jargon_evidence((*term)->id, 1)->front().snippet == "片段11",
            "the newest evidence is kept");
    require(store->jargon_chats((*term)->id)->size() == 2, "chats it was seen in");
    auto counted_only = store->add_jargon_hit("xswl", a, "片段", 100, 0);
    require(counted_only.has_value() && store->jargon_evidence(counted_only->id, 0)->empty(), "no evidence is written when none is kept");
    require(store->set_jargon_inference((*term)->id, "最强一档", 1, 12, false).has_value() &&
                (*store->jargon_by_content("T0"))->meaning == "最强一档",
            "inference stored");
    require(store->set_jargon_manual("T0", "人工的", 200).has_value() &&
                store->set_jargon_inference((*term)->id, "AI 的", 1, 13, false).has_value() &&
                (*store->jargon_by_content("T0"))->meaning == "人工的",
            "a manual meaning wins over inference");
    require(store->delete_jargon("T0").has_value() && !*store->jargon_by_content("T0"), "delete");

    std::mt19937 rng(3);
    std::vector<store::Expression> pool;
    for (int i = 0; i < 30; ++i) pool.push_back({.id = i, .count = i < 12 ? 2 : 1});
    const auto sample = chat::sample_fallback(pool, rng);
    require(sample.size() == 10 && std::set<i64>(std::from_range, sample | std::views::transform(&store::Expression::id)).size() == 10,
            "five weighted plus five repeated, no duplicates");
    require(chat::sample_fallback({}, rng).empty(), "empty pool");
}

void test_selection() {
    require(learn::parse_picks({tool_call("select_expressions", R"({"ids": ["1", "1"]})")}, "select_expressions", "ids", 1, 5) ==
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

std::string joined(const auto &words) {
    std::string out;
    for (const auto &word : words) {
        if (!out.empty()) out += '/';
        out += word;
    }
    return out;
}

/// Expected cuts come from Python jieba 0.42.1 (jieba.cut, default mode).
void test_segment() {
    const std::vector<std::pair<std::string_view, std::string_view>> cases{
        {"我来到北京清华大学", "我/来到/北京/清华大学"},
        {"他来到了网易杭研大厦", "他/来到/了/网易/杭研/大厦"},
        {"小明硕士毕业于中国科学院计算所，后在日本京都大学深造", "小明/硕士/毕业/于/中国科学院/计算所/，/后/在/日本京都大学/深造"},
        {"这卡太T0了 xswl", "这卡/太/T0/了/ /xswl"},
        {"笑死 这也行😂", "笑/死/ /这/也/行/😂"},
        {"iPhone15的价格是5999.99元，涨了3.5%", "iPhone15/的/价格/是/5999.99/元/，/涨/了/3.5%"},
        {"C++和c#都能写，A-B测试", "C++/和/c#/都/能/写/，/A/-/B/测试"},
        {"一个emoji🐱和全角，标点！", "一个/emoji/🐱/和/全角/，/标点/！"},
        {"第一行\r\n第二行", "第一行/\r\n/第二行"},
        {"你怎么这么区，这个区别", "你/怎么/这么/区/，/这个/区别"},
        {"你真菜 你也太区了吧", "你/真菜/ /你/也/太区/了/吧"},
    };
    for (const auto &[text, expected] : cases) {
        require(joined(learn::cut(text)) == expected, std::format("cut {}: {}", text, joined(learn::cut(text))));
    }
    require(learn::cut("").empty(), "nothing to cut");
}

/// Expected terms and tallies come from MaiBot's message_word_frequency_service.py.
void test_frequency() {
    const auto terms = learn::meaningful_terms("我用 C++ 和 node.js 写了个机器人 https://x.com/a @小明 [CQ:face,id=1]");
    require(joined(terms) == "我用/c++/node/js/机器人/node.js",
            "noise removed, stop words and single characters dropped, tech terms added: " + joined(terms));
    require(joined(learn::term_phrases(terms)) == "c++ node/node js/c++ node js", "phrases of 2 and 3 tokens");
    const auto chat = learn::meaningful_terms("这卡太T0了 xswl 哈哈哈 123 ！！");
    require(joined(chat) == "这卡/t0/xswl" && joined(learn::term_phrases(chat)) == "t0 xswl", "laughter, numbers and punctuation dropped");
    require(learn::normalize_term("  《Hello   World》 ") == "hello world", "token normalisation");
    require(learn::normalize_match_text(" XSWL  好耶\n ") == "xswl 好耶", "match normalisation");
    const auto tallies = learn::count_terms({"白龙卡组真好玩", "白龙卡组又赢了", "今天打白龙", "C++ 写机器人", "c++ 写 bot"});
    require(tallies.size() == 2 && tallies[0].term == "白龙卡组" && tallies[0].count == 2 && tallies[0].message_count == 2 &&
                tallies[1].term == "c++" && tallies[1].count == 2,
            "terms seen twice or more, longer first on ties");

    auto store = store::Store::open_memory();
    require(store.has_value(), "open");
    const store::ChatKey a{.type = "group", .id = 1};
    const store::ChatKey b{.type = "group", .id = 2};
    require(store
                ->merge_frequent_terms(
                    a, {{.term = "白龙", .count = 3, .message_count = 2}, {.term = "卡组", .count = 3, .message_count = 3}}, 10, 2)
                .has_value(),
            "merge");
    require(store
                ->merge_frequent_terms(a, {{.term = "白龙", .count = 2, .message_count = 2}, {.term = "x", .count = 1, .message_count = 1}},
                                       20, 2)
                .has_value(),
            "merge again");
    require(store->merge_frequent_terms(b, {{.term = "白龙", .count = 9, .message_count = 9}}, 20, 2).has_value(), "another chat");
    const auto kept = store->frequent_terms(a);
    require(kept.has_value() && kept->size() == 2, "capped at max_terms");
    require((*kept)[0].term == "白龙" && (*kept)[0].occurrence_count == 5 && (*kept)[0].message_count == 4 && (*kept)[0].rank == 1 &&
                (*kept)[0].created_at == 10 && (*kept)[0].updated_at == 20,
            "counts add up; ranked by occurrences");
    require((*kept)[1].term == "卡组" && (*kept)[1].rank == 2 && (*kept)[0].frequency == 5.0 / 8.0 &&
                (*kept)[1].message_frequency == 3.0 / 7.0,
            "the weakest term is dropped; frequencies are shares of the kept terms");
    require(store->frequent_terms(b)->front().occurrence_count == 9, "chats are separate");
}

void test_learned_span() {
    auto store = store::Store::open_memory();
    require(store.has_value() && store->learned_span(7)->has_value() == false, "no span yet");
    require(store->extend_learned_span(7, 100, 200).has_value(), "first span");
    require(store->extend_learned_span(7, 150, 300).has_value() && store->extend_learned_span(7, 50, 60).has_value(), "widen");
    const auto span = store->learned_span(7);
    require(span.has_value() && *span && (*span)->first_time == 50 && (*span)->last_time == 300, "span only widens");
    require(!*store->learned_span(8), "groups are separate");
}

} // namespace

int main() {
    try {
        test_request();
        test_parse();
        test_grounding();
        test_review();
        test_jargon();
        test_sharing();
        test_learnable_text();
        test_store();
        test_selection();
        test_segment();
        test_frequency();
        test_learned_span();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
