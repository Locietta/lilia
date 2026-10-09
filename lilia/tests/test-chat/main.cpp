#include <cstdio>
#include <string>
#include <variant>
#include <vector>

#include <lilia/chat/context.h>
#include <lilia/chat/delivery.h>
#include <lilia/chat/flow.h>
#include <lilia/chat/trigger.h>
#include <lilia/chat/typos.h>
#include <lilia/learn/segment.h>
#include <lilia/onebot/message.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;

constexpr i64 k_self = 10001;
constexpr i64 k_group = 555;

onebot::MessageEvent group_message(onebot::Message message, i64 user = 20002) {
    return {
        .message_type = "group",
        .message_id = 1,
        .user_id = user,
        .group_id = k_group,
        .sender = {.user_id = user, .nickname = "魔王"},
        .message = std::move(message),
    };
}

void test_triggers() {
    ChatConfig config{.groups = {k_group}, .private_users = {30003}, .aliases = {"Lilia", "莉莉娅"}};

    using onebot::at_segment;
    using onebot::text_segment;
    using chat::TriggerReason;

    require(chat::classify(group_message({at_segment(k_self), text_segment(" hi")}), k_self, false, config) == TriggerReason::MENTION,
            "mention");
    require(chat::classify(group_message({text_segment("lilia 在吗")}), k_self, false, config) == TriggerReason::ALIAS,
            "ASCII aliases match case-insensitively");
    require(chat::classify(group_message({text_segment("问问莉莉娅")}), k_self, false, config) == TriggerReason::ALIAS, "CJK alias");
    require(chat::classify(group_message({text_segment("随便聊聊")}), k_self, true, config) == TriggerReason::REPLY, "reply to self");
    require(chat::classify(group_message({text_segment("随便聊聊")}), k_self, false, config) == TriggerReason::NONE, "plain chatter");
    require(chat::classify(group_message({at_segment(k_self)}, k_self), k_self, false, config) == TriggerReason::NONE,
            "own messages never trigger");

    auto other_group = group_message({at_segment(k_self)});
    other_group.group_id = 999;
    require(!chat::chat_allowed(other_group, config), "unlisted groups are not allowed");
    require(chat::classify(other_group, k_self, false, config) == TriggerReason::NONE, "unlisted groups never trigger");

    onebot::MessageEvent dm{.message_type = "private", .user_id = 30003, .message = {text_segment("hello")}};
    require(chat::classify(dm, k_self, false, config) == TriggerReason::PRIVATE, "allowed private chat");
    dm.user_id = 40004;
    require(chat::classify(dm, k_self, false, config) == TriggerReason::NONE, "unlisted private chat");

    std::string_view arguments;
    require(chat::parse_admin_command("  /lilia status ", arguments) && arguments == "status", "admin command");
    require(chat::parse_admin_command("/lilia", arguments) && arguments.empty(), "bare admin command");
    require(!chat::parse_admin_command("/liliastatus", arguments), "prefix must be a whole word");
}

store::StoredMessage stored(i64 id, i64 time, std::string name, std::string text, bool self = false) {
    return {.chat = {.type = "group", .id = k_group},
            .message_id = id,
            .sender_id = self ? k_self : 20000 + id,
            .sender_name = std::move(name),
            .time = time,
            .text = std::move(text),
            .from_self = self};
}

std::string text_of(const provider::Item &item) {
    std::string out;
    for (const auto &part : item.parts) {
        if (const auto *text = std::get_if<provider::TextPart>(&part)) out += text->text;
    }
    return out;
}

void test_time() {
    // 2026-10-03 13:05:00 UTC = 21:05 at UTC+8.
    const i64 t = 1'791'032'700;
    require(chat::format_local_time(t, 480) == "2026-10-03 21:05", "local time: " + chat::format_local_time(t, 480));
    require(chat::local_day_start(t, 480) == 1'790'956'800, "local midnight at UTC+8");
    require(chat::local_day_start(t, 0) == 1'790'985'600, "midnight at UTC");
}

chat::ChatInfo group_info() {
    return {.chat = {.type = "group", .id = k_group}, .chat_name = "测试群", .name = "莉莉娅", .self_id = k_self, .self_name = "闭锁"};
}

provider::ToolCallItem tool_call(std::string name, std::string json) {
    provider::ToolCallItem item{.call = {.id = "toolu_1", .name = std::move(name)}};
    require(!glz::read_json(item.call.input, json), "test tool input must parse");
    return item;
}

void test_planner() {
    const i64 t = 1'791'032'700;
    const auto request = chat::build_planner_request({
        .prompt = "PLANNER",
        .info = group_info(),
        .recent = {stored(101, t - 300, "a", std::string(60, 'x')), stored(102, t - 240, "b", "second"),
                   stored(103, t - 180, "闭锁", "我在", true), stored(104, t - 60, "c", "有人看新番吗")},
        .fresh = {104},
        .addressed = false,
        .now = t,
        .log_bytes = 110,
    });
    require(request.history.size() == 2 && text_of(request.history[0]) == "PLANNER", "planner prompt is the system prompt");
    const auto user = text_of(request.history[1]);
    require(user.find(std::string(60, 'x')) == std::string::npos, "the oldest line is dropped first when over budget");
    require(request.ids == std::vector<i64>({102, 103, 104}), "ids map #n to message ids in order");
    require(user.find("#3 [21:04] c(20104): 有人看新番吗") != std::string::npos, "lines are numbered");
    require(user.find("以下是新消息") < user.find("#3 "), "the new messages are marked");
    require(user.find("以下是新消息") > user.find("#2 "), "old lines come before the marker");
    require(user.find("没有人直接叫莉莉娅") != std::string::npos, "the planner is told nobody addressed her");
    require(request.tools.size() == 2 && request.tools[0].name == "reply" && request.tools[1].name == "stay_silent", "two tools");

    using provider::OutputItem;
    const auto reply = chat::parse_plan(
        {OutputItem(
            tool_call("reply", R"({"target":"#3 c: 有人看新番吗","quote":true,"style":"normal","search":false,"intent":"安利一部"})"))},
        request.ids);
    require(reply.reply && reply.target == 104 && reply.quote && reply.style == chat::ReplyStyle::NORMAL && reply.intent == "安利一部",
            "reply arguments are read; a target with trailing text still parses");
    const auto silent = chat::parse_plan({OutputItem(tool_call("stay_silent", R"({"reason":"和她无关"})"))}, request.ids);
    require(!silent.reply && silent.reason == "和她无关", "stay_silent carries its reason");
    require(!chat::parse_plan({OutputItem(tool_call("reply", R"({"target":"#9","style":"short","intent":"x"})"))}, request.ids).reply,
            "an out-of-range target is silence");
    require(!chat::parse_plan({}, request.ids).reply, "no tool call is silence");

    auto followup_input = chat::PlannerInput{
        .prompt = "P", .info = group_info(), .recent = {stored(401, t, "c", "a"), stored(402, t, "d", "b")}, .fresh = {401, 402}};
    followup_input.decisions = {"[21:04] 没说话：别人在私聊", "[21:05] 回复了c的「a」：接一句"};
    followup_input.just_replied = 401;
    const auto followup = text_of(chat::build_planner_request(followup_input).history[1]);
    require(followup.find("她刚回复了 #1") != std::string::npos && followup.find("没有人直接叫") == std::string::npos,
            "the look after a reply says what she answered");
    require(followup.find("- [21:05] 回复了c的「a」：接一句") != std::string::npos && followup.find("别重复") != std::string::npos,
            "recent decisions are listed");

    auto reference_input =
        chat::PlannerInput{.prompt = "P", .info = group_info(), .recent = {stored(301, t, "c", "看新番吗")}, .fresh = {301}};
    reference_input.reference_handoff = true;
    const auto reference_request = chat::build_planner_request(reference_input);
    const auto &reply_schema = reference_request.tools[0].input_schema;
    require(std::ranges::contains(reply_schema.required, std::string("brief")) &&
                !std::ranges::contains(reply_schema.required, std::string("intent")),
            "the reference handoff asks for brief instead of intent");
    const auto briefed = chat::parse_plan(
        {OutputItem(tool_call("reply", R"({"target":"#1","style":"short","brief":"在聊新番，她最近在追"})"))}, reference_request.ids);
    require(briefed.reply && briefed.brief == "在聊新番，她最近在追", "brief is read");
    require(chat::handoff_rule(true).find("brief") != std::string_view::npos &&
                chat::handoff_rule(false).find("intent") != std::string_view::npos,
            "the planner rule follows the handoff");
}

void test_writer() {
    const i64 t = 1'791'032'700;
    const auto history = chat::build_writer_history({
        .persona = "PERSONA",
        .rules = "RULES",
        .info = group_info(),
        .recent = {stored(201, t - 300, "闭锁", "早", true), stored(202, t - 240, "b", "第三集作画崩了"), stored(203, t - 200, "c", "确实"),
                   stored(204, t - 180, "闭锁", "崩成ppt了", true), stored(205, t - 60, "b", "莉莉娅你追吗")},
        .target = stored(205, t - 60, "b", "莉莉娅你追吗"),
        .plan = {.reply = true, .target = 205, .style = chat::ReplyStyle::SHORT, .intent = "说在追"},
        .now = t,
        .log_bytes = 10'000,
    });
    using provider::Role;
    require(history.size() == 7, "system, developer, then user/assistant turns");
    require(history[0].role == Role::SYSTEM && text_of(history[0]) == "PERSONA", "persona is the system prompt");
    require(history[1].role == Role::DEVELOPER && text_of(history[1]).find("RULES") != std::string::npos &&
                text_of(history[1]).find("一两个词") != std::string::npos,
            "rules and the style instruction are developer text");
    require(history[2].role == Role::USER && text_of(history[2]) == "（更早的聊天记录略）",
            "a log starting with her own line gets a user stub");
    require(history[3].role == Role::ASSISTANT && text_of(history[3]) == "早", "her lines are assistant turns without labels");
    require(history[4].role == Role::USER && text_of(history[4]) == "b: 第三集作画崩了\nc: 确实", "consecutive lines share a turn");
    require(history[5].role == Role::ASSISTANT && text_of(history[5]) == "崩成ppt了", "assistant turn");
    const auto last = text_of(history[6]);
    require(history[6].role == Role::USER && last.starts_with("b: 莉莉娅你追吗\n\n【你要回应b的这条") && last.ends_with("现在，你说："),
            "the note ends the last user turn");
    require(last.find("说在追") != std::string::npos, "the intent reaches the writer");
    const auto briefed = chat::build_writer_history({
        .persona = "PERSONA",
        .rules = "RULES",
        .info = group_info(),
        .recent = {stored(205, t - 60, "b", "莉莉娅你追吗")},
        .target = stored(205, t - 60, "b", "莉莉娅你追吗"),
        .plan = {.reply = true, .target = 205, .brief = "b 在问她追不追第三集"},
        .now = t,
    });
    const auto briefed_note = text_of(briefed.back());
    require(briefed_note.find("回复信息参考：b 在问她追不追第三集") != std::string::npos &&
                briefed_note.find("不用完全遵守") != std::string::npos && briefed_note.find("你想表达的意思") == std::string::npos,
            "a brief reaches the writer as optional reference");
    require(text_of(history[1]).find("口气") == std::string::npos, "no voice section without samples");
}

void test_flow() {
    chat::FlowSettings settings{.talk_value = 0.25, .window_seconds = 300};
    chat::Flow flow;
    require(!chat::should_plan(flow, settings, 1000), "nothing to plan");
    for (i64 i = 1; i <= 3; ++i) flow.unseen.push_back(stored(i, 1000, "a", "x"));
    require(!chat::should_plan(flow, settings, 1000), "below ceil(1/talk_value) messages");
    flow.unseen.push_back(stored(4, 1000, "a", "x"));
    require(chat::should_plan(flow, settings, 1000), "four messages at talk_value 0.25");

    flow.unseen.resize(1);
    flow.addressed = 1;
    require(chat::should_plan(flow, settings, 1000), "being addressed always plans");
    flow.addressed.reset();

    chat::note_spoke(flow, 1000);
    require(chat::should_plan(flow, settings, 1200), "inside the conversation window every message counts");
    require(!chat::should_plan(flow, settings, 1400), "the window closes");

    chat::note_silent(flow, 2000);
    require(flow.backoff_until == 0, "one silent decision does not back off");
    chat::note_silent(flow, 2000);
    require(flow.backoff_until == 2015, "two in a row back off 15s");
    chat::note_silent(flow, 2000);
    require(flow.backoff_until == 2030, "the backoff doubles");
    flow.unseen.assign(5, stored(9, 2000, "a", "x"));
    require(!chat::should_plan(flow, settings, 2010), "backoff holds below the burst size");
    flow.unseen.push_back(stored(10, 2000, "a", "x"));
    require(chat::should_plan(flow, settings, 2010), "a burst bypasses the backoff");
    chat::note_spoke(flow, 2020, 20001);
    require(flow.silent_streak == 0 && flow.backoff_until == 0, "speaking resets the backoff");
    chat::note_silent(flow, 2030);
    chat::note_silent(flow, 2030);
    flow.unseen.assign(1, stored(5, 2031, "a", "x"));
    require(!chat::should_plan(flow, settings, 2031), "others are held back by the backoff");
    flow.unseen.push_back(stored(1, 2031, "b", "你是gpt吗"));
    require(chat::should_plan(flow, settings, 2031), "whom she just answered is not");
    require(!chat::should_plan(flow, settings, 2400), "only inside the conversation window");

    chat::FlowSettings shy{.talk_value = 0};
    chat::Flow quiet;
    quiet.unseen.assign(20, stored(1, 1000, "a", "x"));
    require(!chat::should_plan(quiet, shy, 1000), "talk_value 0 only reacts when addressed");
}

std::vector<std::string> texts(const std::vector<chat::Segment> &segments) {
    std::vector<std::string> out;
    for (const auto &segment : segments) out.push_back(segment.text);
    return out;
}

/// Expected pieces come from MaiBot's _split_into_sentence_segments with
/// merging switched off.
void test_bubbles() {
    using chat::Segment;
    require(chat::split_pieces("好耶。\n我也在追") == std::vector<Segment>({{"好耶", "。"}, {"我也在追", ""}}),
            "marks split and are dropped");
    require(texts(chat::split_pieces("骂的又不是你，圈内撕版权和抢饭碗呗，个人小项目拿它当草稿参考又没啥")) ==
                std::vector<std::string>({"骂的又不是你", "圈内撕版权和抢饭碗呗", "个人小项目拿它当草稿参考又没啥"}),
            "commas split");
    require(texts(chat::split_pieces("独立游戏小成本用 AI 出图挺好的，真不用太在意那些骂声")) ==
                std::vector<std::string>({"独立游戏小成本用", "AI", "出图挺好的", "真不用太在意那些骂声"}),
            "spaces split unless between Latin letters or digits");
    require(chat::split_pieces("他说：\"别急，等会儿\"然后就走了").size() == 1, "not inside quotes");
    require(texts(chat::split_pieces("时间是 10:30, 别迟到")) == std::vector<std::string>({"时间是", "10:30", "别迟到"}), "colons");
    require(chat::split_pieces("use godot for a small demo").size() == 1, "English stays whole");
    require(texts(chat::split_pieces("这个 - 那个 好吧")) == std::vector<std::string>({"这个 - 那个", "好吧"}), "not beside a dash");
    require(chat::split_pieces("“一，二”三，四").size() == 1, "a curly quote never closes (as in MaiBot)");
    require(chat::split_pieces("  \n ").empty() && chat::split_pieces("好").size() == 1, "blank text; short text stays whole");

    // Merging only joins neighbours back together with their separator.
    for (unsigned seed = 0; seed < 50; ++seed) {
        std::mt19937 rng(seed);
        const auto merged = chat::split_segments("骂的又不是你，圈内撕版权和抢饭碗呗，个人小项目拿它当草稿参考又没啥", rng);
        std::string joined;
        for (const auto &segment : merged) joined += segment.text + segment.separator;
        require(joined == "骂的又不是你，圈内撕版权和抢饭碗呗，个人小项目拿它当草稿参考又没啥" && !merged.empty() && merged.size() <= 3,
                "merging keeps the text");
    }

    const std::vector<Segment> six{{"一", "，"}, {"二", "，"}, {"三", "，"}, {"四", "，"}, {"五", "，"}, {"六", ""}};
    require(texts(chat::pack_segments(six, 3)) == std::vector<std::string>({"一，二", "三，四", "五，六"}), "packed evenly");
    auto corrected = six;
    corrected[5].quote_previous = true;
    const auto packed = chat::pack_segments(corrected, 3);
    require(texts(packed) == std::vector<std::string>({"一，二", "三，四，五", "六"}) && packed[2].quote_previous,
            "a correction keeps its own message");
    require(chat::pack_segments(six, 8).size() == 6, "few enough: unchanged");
    require(chat::whole_message(" 第一段很长，很长很长很长很长很长很长很长。\n\n第二段 \n") ==
                std::vector<std::string>({"第一段很长，很长很长很长很长很长很长很长。\n第二段"}),
            "a long reply stays one message with its lines");
    require(chat::whole_message(" \n ").empty(), "nothing to send");
    require(chat::first_lines("一\n\n 二 \n三", 2) == "一\n二", "first lines, blanks skipped");

    require(chat::fit_length("短的就不动", 30) == "短的就不动", "short text is untouched");
    require(chat::fit_length("这集作画不错，打戏很顶。剧情也推进了！下周还看", 20) == "这集作画不错，打戏很顶。剧情也推进了！",
            "whole sentences that fit are kept");
    require(chat::fit_length("这集作画不错，打戏很顶，剧情也推进了，下周还看", 12) == "这集作画不错，打戏很顶，剧情也推进了，下周还看",
            "a sentence is never cut, even past the limit");
    require(chat::fit_length("小成本用 AI 出图挺好的", 10) == "小成本用 AI 出图挺好的", "not even at a space");
    require(chat::fit_length("真的吗？？那我也去", 3) == "真的吗？？", "a run of marks stays with its sentence");
    require(chat::fit_length("第一行\n第二行很长很长很长", 6) == "第一行", "newlines are breaks too");
    require(chat::fit_length("刚手滑了。其实我想说的是，想法挺好，做起来难", 15) == "刚手滑了。", "a cut ends on a whole sentence");
    require(chat::strip_asides("好耶（笑）真不错(指C++)和(C++)") == "好耶真不错和(C++)", "CJK asides are removed, code stays");
    require(chat::strip_asides("没写完（笑") == "没写完（笑", "an unclosed bracket stays");
    require(chat::strip_asides("其实——也还行") == "其实，也还行", "a dash becomes a comma");

    require(chat::typing_delay("在", 1.0) == std::chrono::milliseconds(1200), "a lone character: 1.2s");
    require(chat::typing_delay("好耶ok", 1.0) == std::chrono::milliseconds(900), "0.3s per Han character, 0.15s per other");
    require(chat::typing_delay("好耶ok", 2.0) == std::chrono::milliseconds(450), "faster typing");
    require(chat::typing_delay("在", 0.0) == std::chrono::milliseconds(0), "speed 0 sends immediately");
}

/// Expected candidates come from MaiBot's ChineseTypoGenerator.
void test_typos() {
    require(chat::reading_of(U'做') == "zuo4" && chat::readings_of("银行") == std::vector<std::string>({"yin2", "hang2"}),
            "readings, in context for dictionary words");
    require(chat::similar_chars(U'做', "zuo4", "", 9) == std::vector<char32_t>({U'作', U'坐', U'座'}),
            "frequent homophones, likeliest first");
    require(chat::similar_chars(U'在', "zai4", "", 9).empty(), "much rarer homophones are never used");
    require(chat::similar_chars(U'在', "zai4", "zai3", 9).empty() && !chat::similar_chars(U'做', "zuo4", "zuo3", 9).empty(),
            "another tone adds candidates");
    require(chat::word_homophones("事实", 9) == std::vector<std::string>({"适时"}) && chat::word_homophones("时候", 9).empty(),
            "homophone words from the dictionary");
    require(std::abs(chat::char_frequency(U'做') - 60.577678) < 1e-5, "character frequency from jieba's dictionary");

    lilia::TypoConfig always{.error_rate = 1.0, .tone_error_rate = 0.0, .word_replace_rate = 0.0};
    int changed = 0;
    for (unsigned seed = 0; seed < 40; ++seed) {
        std::mt19937 rng(seed);
        const auto typoed = chat::make_typos("我来做饭", always, rng);
        const auto before = learn::utf8_chars("我来做饭");
        const auto after = learn::utf8_chars(typoed.text);
        require(after.size() == before.size(), "one character for one: " + typoed.text);
        for (usize i = 0; i < before.size(); ++i) {
            if (after[i].cp == before[i].cp) continue;
            const auto allowed = chat::similar_chars(before[i].cp, chat::reading_of(before[i].cp), "", 9);
            require(std::ranges::contains(allowed, after[i].cp), "only homophone candidates replace characters: " + typoed.text);
        }
        if (typoed.text != "我来做饭") ++changed;
        require(typoed.correction.empty() || typoed.text != "我来做饭", "a correction only for a typo");
    }
    require(changed > 0, "typos happen at error rate 1");
    lilia::TypoConfig never{.error_rate = 0.0, .word_replace_rate = 0.0};
    std::mt19937 rng(1);
    require(chat::add_typos({{"我来做饭", "，"}}, never, rng) == std::vector<chat::Segment>({{"我来做饭", "，"}}), "no typos at rate 0");
    lilia::TypoConfig off{.enabled = false, .error_rate = 1.0};
    require(chat::add_typos({{"我来做饭", ""}}, off, rng).front().text == "我来做饭", "disabled");
    bool quoted = false;
    for (unsigned seed = 0; seed < 200 && !quoted; ++seed) {
        std::mt19937 r(seed);
        const auto out = chat::add_typos({{"我来做饭", ""}}, always, r);
        if (out.size() == 2) quoted = out[1].quote_previous && out[1].text.size() == std::string("做").size();
    }
    require(quoted, "a correction follows its typo, quoting it");
}
void test_clean_reply() {
    const chat::ChatInfo info{.self_id = k_self, .self_name = "lilia"};
    const std::vector<std::string> aliases{"莉莉娅"};
    require(chat::clean_reply("  好的呀\n", info, aliases, 100) == "好的呀", "trim");
    require(chat::clean_reply("[21:05] lilia(你): 在的", info, aliases, 100) == "在的", "copied log prefix is stripped");
    require(chat::clean_reply("莉莉娅：嗯嗯", info, aliases, 100) == "嗯嗯", "full-width colon label is stripped");
    require(chat::clean_reply("lilia 很开心", info, aliases, 100) == "lilia 很开心", "a name without a colon is content");
    require(chat::clean_reply("@阿紫 芒果\n昨天说的", info, aliases, 100) == "芒果\n昨天说的", "a typed leading mention is dropped");
    require(chat::clean_reply("邮箱是 a@b.c", info, aliases, 100) == "邮箱是 a@b.c", "an @ inside the text stays");
    require(chat::strip_markdown("## 结论\n**26.3** 是最新版，a**b") == "结论\n26.3 是最新版，ab", "headings and bold are stripped");
    require(chat::strip_markdown("#话题 不是标题") == "#话题 不是标题", "a hash without a space is content");
    const std::vector<store::StoredMessage> recent{stored(1, 0, "Locietta", "@莉莉娅 玩游戏吗？"), stored(2, 0, "闭锁", "啥", true)};
    require(chat::strip_transcript("Locietta: 啥游戏", recent) == "啥游戏", "another speaker's label is stripped");
    require(chat::strip_transcript("Locietta: @莉莉娅 玩游戏吗？\n10001:", recent).empty(), "an echoed transcript is dropped");
    require(chat::strip_transcript("@莉莉娅 玩游戏吗？\n12:30 开始", recent) == "@莉莉娅 玩游戏吗？\n12:30 开始",
            "unlabelled lines and short numbers are content");
    const auto bounded = chat::clean_reply(std::string(30, 'a') + "你好你好", info, aliases, 20);
    require(bounded.size() <= 20 && bounded.ends_with("……"), "long replies are bounded with an ellipsis");
}

} // namespace

int main() {
    try {
        test_triggers();
        test_time();
        test_planner();
        test_writer();
        test_flow();
        test_bubbles();
        test_typos();
        test_clean_reply();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
