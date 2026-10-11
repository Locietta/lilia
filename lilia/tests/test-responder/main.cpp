#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <lighter/async/io/loop.h>
#include <lighter/async/runtime/task_group.h>

#include <lilia/chat/context.h>
#include <lilia/chat/responder.h>
#include <lilia/onebot/message.h>
#include <lilia/tests/support/fake_provider.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;
using lighter::Task;

constexpr i64 k_self = 10001;
constexpr i64 k_group = 555;
const store::ChatKey k_chat{.type = "group", .id = k_group};

std::string text_of(const provider::Item &item) {
    std::string out;
    for (const auto &part : item.parts) {
        if (const auto *text = std::get_if<provider::TextPart>(&part)) out += text->text;
        if (const auto *outcome = std::get_if<ToolOutcome>(&part)) out += outcome->receipt;
    }
    return out;
}

/// True once the planner has seen the result of its own call this turn.
bool after_tool_result(const provider::History &history) {
    return !history.empty() && !history.back().parts.empty() && std::holds_alternative<ToolOutcome>(history.back().parts.front());
}

/// "#n" of the first line of the planner's log containing `needle`.
std::string line_number(const provider::History &history, std::string_view needle) {
    for (const auto &item : history) {
        const auto text = text_of(item);
        for (usize start = 0; start < text.size();) {
            const auto end = text.find('\n', start);
            const auto line = std::string_view(text).substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (line.starts_with('#') && line.contains(needle)) return std::string(line.substr(0, line.find(' ')));
            if (end == std::string::npos) break;
            start = end + 1;
        }
    }
    return "#0";
}

/// A Responder on fake models and a fake clock, sending into `sent`.
struct Harness {
    Config config;
    store::Store store;
    std::shared_ptr<test::FakeProvider::State> planner = std::make_shared<test::FakeProvider::State>();
    std::shared_ptr<test::FakeProvider::State> writer = std::make_shared<test::FakeProvider::State>();
    std::vector<std::string> sent;
    i64 clock = 1'791'032'700;
    i64 next_sent = 900;
    std::unique_ptr<chat::Responder> responder;

    explicit Harness(lighter::TaskGroup<> &tasks) : store(std::move(*store::Store::open_memory())) {
        config.chat.groups = {k_group};
        config.chat.typos.enabled = false;
        config.chat.alt_style_probability = 0;
        config.learning.enabled = false;
        config.model.web_search = false;
        responder = std::make_unique<chat::Responder>(config, store, test::fake_provider(planner), test::fake_provider(writer),
                                                      test::fake_provider(writer), "missing-persona.md",
                                                      [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
        responder->realtime = false;
        responder->self = {.user_id = k_self, .nickname = "lilia"};
        responder->now = [this] { return clock; };
        responder->next_notice_id = -1;
        responder->outbox = [this](const store::ChatKey &, onebot::Message message) -> Task<std::optional<i64>> {
            sent.push_back(onebot::render_plain(message));
            co_return ++next_sent;
        };
    }

    std::optional<store::ChatKey> say(i64 id, i64 user, std::string name, onebot::Message message) {
        clock += 5;
        return responder->ingest({
            .time = clock,
            .self_id = k_self,
            .message_type = "group",
            .message_id = id,
            .user_id = user,
            .group_id = k_group,
            .group_name = "测试群",
            .sender = {.user_id = user, .nickname = std::move(name)},
            .message = std::move(message),
        });
    }

    std::string stored_text(i64 id) {
        auto found = store.message(k_chat, id);
        return found.has_value() && *found ? (*found)->text : std::string("<missing>");
    }

    std::vector<std::string> notices() {
        std::vector<std::string> out;
        for (const auto &message : *store.recent_messages(k_chat, 50)) {
            if (chat::is_notice(message)) out.push_back(message.text);
        }
        return out;
    }
};

Task<> test_quotes_and_recalls() {
    lighter::TaskGroup tasks(lighter::EventLoop::current());
    Harness h(tasks);
    std::ignore = h.say(1, 20001, "小明", {onebot::text_segment("今晚打吗")});
    std::ignore = h.say(2, 20002, "小红", {onebot::reply_segment(1), onebot::text_segment("打")});
    require(h.stored_text(2) == "[回复 小明: 今晚打吗] 打", "a quote is stored with what it quotes: " + h.stored_text(2));
    std::ignore = h.say(3, 20002, "小红", {onebot::reply_segment(77), onebot::text_segment("?")});
    require(h.stored_text(3) == "[回复] ?", "an unknown quote");

    // Called, then the message is taken back: it is no longer hers to answer.
    require(h.say(4, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" 你几点睡")}).has_value(), "she is called");
    h.responder->recall(k_chat, 4);
    require(h.stored_text(4) == onebot::k_recalled_text, "the recall is stored");
    auto &flow = h.responder->chats[k_chat].flow;
    require(!flow.addressed && std::ranges::none_of(flow.unseen, [](const auto &m) { return m.message_id == 4; }),
            "nor does it call her any more");

    // The planner still tries to answer it; the receipt says why not.
    std::ignore = h.say(5, 20002, "小红", {onebot::text_segment("嗯")});
    h.planner->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        if (after_tool_result(history)) return std::vector{test::tool_call("stay_silent", R"({"reason":"算了"})")};
        const auto target = line_number(history, onebot::k_recalled_text);
        return std::vector{test::tool_call("reply", R"({"target":")" + target + R"(","style":"short","intent":"回一句"})")};
    };
    co_await h.responder->run_turn(k_chat);
    require(h.sent.empty() && h.writer->requests.empty(), "nothing was written for a recalled message");
    require(h.planner->requests.size() == 2 && text_of(h.planner->requests[1].back()).contains("撤回"), "the planner learns why");
    co_await tasks.join();
}

Task<> test_mutes_and_notices() {
    lighter::TaskGroup tasks(lighter::EventLoop::current());
    Harness h(tasks);
    std::ignore = h.say(1, 20001, "小明", {onebot::text_segment("今晚打吗")});
    std::ignore = h.say(2, 20004, "管理", {onebot::text_segment("别刷了")});
    h.responder->notice({.time = h.clock,
                         .notice_type = "group_ban",
                         .sub_type = "ban",
                         .group_id = k_group,
                         .user_id = k_self,
                         .operator_id = 20004,
                         .duration = 600});
    require(!h.say(3, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" 在吗")}).has_value(), "muted, she is not asked");
    require(h.responder->chats[k_chat].flow.unseen.empty() && !h.responder->chats[k_chat].flow.addressed, "nothing waits for later");
    co_await h.responder->run_turn(k_chat);
    require(h.planner->requests.empty(), "no model calls while muted");

    h.responder->notice({.notice_type = "group_ban", .sub_type = "lift_ban", .group_id = k_group, .user_id = k_self, .operator_id = 20004});
    h.planner->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        if (after_tool_result(history)) return std::vector{test::tool_call("stay_silent", R"({"reason":"说完了"})")};
        const auto target = line_number(history, "还在吗");
        return std::vector{test::tool_call("reply", R"({"target":")" + target + R"(","style":"short","intent":"说在"})")};
    };
    h.writer->answer = [](const auto &, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        return std::vector{test::text_answer("在的")};
    };
    require(h.say(4, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" 还在吗")}).has_value(), "unmuted, she is asked");
    co_await h.responder->run_turn(k_chat);
    require(h.sent == std::vector<std::string>{"在的"}, "and answers");

    glz::generic_i64 likes;
    require(!glz::read_json(likes, R"([{"emoji_id":"76","count":1}])"), "likes parse");
    h.responder->notice({.time = h.clock,
                         .notice_type = "group_msg_emoji_like",
                         .group_id = k_group,
                         .user_id = 20002,
                         .message_id = 1,
                         .likes = likes,
                         .is_add = true});
    h.responder->notice(
        {.time = h.clock, .notice_type = "notify", .sub_type = "poke", .group_id = k_group, .user_id = 20001, .target_id = 20004});
    h.responder->notice({.time = h.clock, .notice_type = "group_increase", .group_id = k_group, .user_id = 30003});
    h.responder->notice({.time = h.clock, .notice_type = "group_recall", .group_id = 999, .message_id = 1});
    const auto lines = h.notices();
    require(lines == std::vector<std::string>{"管理把lilia禁言了10分钟", "管理解除了lilia的禁言", "20002给小明的「今晚打吗」贴了[赞]",
                                              "小明戳了戳管理", "30003加入了群聊"},
            "notice lines, in order");
    require(h.stored_text(1) == "今晚打吗", "notices of other groups are ignored");
    co_await tasks.join();
}

Task<> test_names_do_not_force() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    h.planner->answer = [](const provider::History &, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        return std::vector{test::tool_call("stay_silent", R"({"reason":"只是在说她"})")};
    };
    h.writer->answer = [](const auto &, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        return std::vector{test::text_answer("嗯？")};
    };
    // Talked about, not to: the planner may let it pass.
    require(h.say(1, 20001, "小明", {onebot::text_segment("莉莉娅好像又坏了")}).has_value(), "her name opens the gate");
    co_await h.responder->run_turn(k_chat);
    require(h.sent.empty() && h.writer->requests.empty(), "and silence stands");
    require(text_of(h.planner->requests[0].back()).contains("提到了莉莉娅的名字"), "the planner knows it may be about her");

    // Called with an @: silence is not an answer.
    require(h.say(2, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" 你坏了吗")}).has_value(), "an @");
    co_await h.responder->run_turn(k_chat);
    require(h.sent == std::vector<std::string>{"嗯？"}, "she answers anyway");
    co_await tasks.join();
}

} // namespace

int main() {
    try {
        for (auto *test : {&test_quotes_and_recalls, &test_mutes_and_notices, &test_names_do_not_force}) {
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
