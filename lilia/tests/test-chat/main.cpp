#include <cstdio>
#include <string>
#include <variant>
#include <vector>

#include <lilia/chat/context.h>
#include <lilia/chat/delivery.h>
#include <lilia/chat/flow.h>
#include <lilia/chat/trigger.h>
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
    chat::note_spoke(flow, 2020);
    require(flow.silent_streak == 0 && flow.backoff_until == 0, "speaking resets the backoff");

    chat::FlowSettings shy{.talk_value = 0};
    chat::Flow quiet;
    quiet.unseen.assign(20, stored(1, 1000, "a", "x"));
    require(!chat::should_plan(quiet, shy, 1000), "talk_value 0 only reacts when addressed");
}

void test_bubbles() {
    require(chat::split_bubbles("好耶。\n\n我也在追") == std::vector<std::string>({"好耶", "我也在追"}), "one bubble per line, no period");
    require(chat::split_bubbles("一\n二\n三\n四") == std::vector<std::string>({"一", "二", "三 四"}), "extra lines merge into the last");
    const std::string paragraph = "这一集的作画确实有点崩，尤其是打斗那段。不过剧情推进得还挺快的！下周应该会好一点吧？反正我会接着看";
    const auto pieces = chat::split_bubbles(paragraph);
    require(pieces.size() == 3 && pieces[0] == "这一集的作画确实有点崩，尤其是打斗那段" && pieces[1] == "不过剧情推进得还挺快的！",
            "a long line splits after sentence ends");
    require(chat::split_bubbles("啊？？真的假的").size() == 1, "short lines stay whole");
    require(chat::split_bubbles("  \n ").empty(), "blank text has no bubbles");

    require(chat::fit_length("短的就不动", 30) == "短的就不动", "short text is untouched");
    require(chat::fit_length("这集作画不错，打戏很顶，剧情也推进了，下周还看", 12) == "这集作画不错，打戏很顶",
            "whole clauses that fit are kept");
    require(chat::fit_length("一二三四五六七八九十", 4) == "一二三四", "an overlong first clause is cut by characters");
    require(chat::fit_length("第一行\n第二行很长很长很长", 6) == "第一行", "newlines are breaks too");
    require(chat::fit_length("工具就是工具 拿来提高效率没问题 最烦的是拿ai图装成自己画的", 18) == "工具就是工具 拿来提高效率没问题",
            "spaces are clause breaks, so nothing is cut mid-word");
    require(chat::strip_asides("好耶（笑）真不错(指C++)和(C++)") == "好耶真不错和(C++)", "CJK asides are removed, code stays");
    require(chat::strip_asides("没写完（笑") == "没写完（笑", "an unclosed bracket stays");
    require(chat::strip_asides("其实——也还行") == "其实，也还行", "a dash becomes a comma");

    std::mt19937 rng(7);
    require(chat::typing_delay("在", 1.0, rng) == std::chrono::milliseconds(600), "short bubbles take the minimum");
    require(chat::typing_delay(std::string(300, 'a'), 1.0, rng) == std::chrono::milliseconds(6000), "long bubbles are capped");
    require(chat::typing_delay("在", 0.0, rng) == std::chrono::milliseconds(0), "speed 0 sends immediately");
}
void test_clean_reply() {
    const chat::ChatInfo info{.self_id = k_self, .self_name = "lilia"};
    const std::vector<std::string> aliases{"莉莉娅"};
    require(chat::clean_reply("  好的呀\n", info, aliases, 100) == "好的呀", "trim");
    require(chat::clean_reply("[21:05] lilia(你): 在的", info, aliases, 100) == "在的", "copied log prefix is stripped");
    require(chat::clean_reply("莉莉娅：嗯嗯", info, aliases, 100) == "嗯嗯", "full-width colon label is stripped");
    require(chat::clean_reply("lilia 很开心", info, aliases, 100) == "lilia 很开心", "a name without a colon is content");
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
        test_clean_reply();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
