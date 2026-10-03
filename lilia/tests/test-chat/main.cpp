#include <cstdio>
#include <string>
#include <variant>
#include <vector>

#include <lilia/chat/context.h>
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

void test_context() {
    const i64 t = 1'791'032'700;
    chat::ContextRequest request{
        .persona = "PERSONA",
        .info = {.chat = {.type = "group", .id = k_group}, .chat_name = "测试群", .self_id = k_self, .self_name = "lilia"},
        .recent = {stored(1, t - 300, "a", std::string(60, 'x')), stored(2, t - 240, "b", "second"),
                   stored(3, t - 180, "lilia", "我在", true), stored(4, t - 60, "c", "@lilia 你好")},
        .triggers = {stored(4, t - 60, "c", "@lilia 你好")},
        .now = t,
        .utc_offset_minutes = 480,
        .log_bytes = 90,
    };
    const auto history = chat::build_context(request);
    require(history.size() == 3, "system, developer and one user turn");
    require(history[0].role == provider::Role::SYSTEM && text_of(history[0]) == "PERSONA", "persona is the system prompt");
    require(history[1].role == provider::Role::DEVELOPER && text_of(history[1]).find("测试群") != std::string::npos,
            "runtime facts name the chat");

    const auto user = text_of(history[2]);
    require(user.find("second") != std::string::npos && user.find("(你): 我在") != std::string::npos, "recent lines are rendered");
    require(user.find(std::string(60, 'x')) == std::string::npos, "the oldest line is dropped first when over budget");
    require(user.find("[21:04] c(20004): @lilia 你好") != std::string::npos, "trigger is rendered with time and id");
    require(user.find("你好") == user.rfind("你好"), "the trigger appears once, not also in the log");
    require(user.find("second") < user.find("我在"), "log stays in chronological order");
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
    const auto bounded = chat::clean_reply(std::string(30, 'a') + "你好你好", info, aliases, 20);
    require(bounded.size() <= 20 && bounded.ends_with("……"), "long replies are bounded with an ellipsis");
}

} // namespace

int main() {
    try {
        test_triggers();
        test_time();
        test_context();
        test_clean_reply();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
