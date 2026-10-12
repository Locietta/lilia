#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
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
#include <lilia/sticker/sticker.h>
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
        if (const auto *outcome = std::get_if<ToolOutcome>(&part)) out += outcome->receipt + "\n";
    }
    return out;
}

/// True when the planner's last call has its result and nobody else has
/// said anything since (only her own lines may follow it).
bool after_tool_result(const provider::History &history) {
    if (history.empty() || history.back().parts.empty() || !std::holds_alternative<ToolOutcome>(history.back().parts.front())) return false;
    for (const auto &part : history.back().parts) {
        const auto *text = std::get_if<provider::TextPart>(&part);
        if (!text) continue;
        for (usize start = 0; start < text->text.size();) {
            const auto end = text->text.find('\n', start);
            const auto line = std::string_view(text->text).substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (line.starts_with('#') && !line.contains("(她自己)") && !line.contains("起是新消息")) return false;
            if (end == std::string::npos) break;
            start = end + 1;
        }
    }
    return true;
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
    std::vector<onebot::Message> outgoing;
    i64 clock = 1'791'032'700;
    i64 next_sent = 900;
    std::unique_ptr<chat::Responder> responder;

    explicit Harness(lighter::TaskGroup<> &tasks) : store(std::move(*store::Store::open_memory())) {
        config.chat.groups = {k_group};
        config.chat.typos.enabled = false;
        config.learning.enabled = false;
        config.model.web_search = false;
        responder = std::make_unique<chat::Responder>(config, store, test::fake_provider(planner), test::fake_provider(writer),
                                                      test::fake_provider(writer), "missing-persona.md",
                                                      [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
        responder->realtime = false;
        // Replies split into bubbles at random; tests see the same split every run.
        responder->rng.seed(1);
        responder->self = {.user_id = k_self, .nickname = "lilia"};
        responder->now = [this] { return clock; };
        responder->next_notice_id = -1;
        responder->outbox = [this](const store::ChatKey &, onebot::Message message) -> Task<std::optional<i64>> {
            sent.push_back(onebot::render_plain(message));
            outgoing.push_back(std::move(message));
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

/// The planner's analysis plus a call, as a model answer.
std::vector<provider::OutputItem> analysed(std::string analysis, provider::OutputItem call) {
    return {test::text_answer(std::move(analysis)), std::move(call)};
}

Task<> test_the_planner_decides() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    h.planner->answer = [](const provider::History &, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        return analysed("只是在说她，不用接", test::tool_call("stay_silent", R"({"reason":"只是在说她"})"));
    };
    // Named, or even @-ed: the planner looks at once, and its silence stands.
    require(h.say(1, 20001, "小明", {onebot::text_segment("莉莉娅好像又坏了")}).has_value(), "her name opens the gate");
    co_await h.responder->run_turn(k_chat);
    require(text_of(h.planner->requests[0].back()).contains("提到了她的名字"), "a fact, not an instruction");
    require(h.say(2, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" 你坏了吗")}).has_value(), "an @");
    co_await h.responder->run_turn(k_chat);
    require(text_of(h.planner->requests[1].back()).contains("是对她说的") && text_of(h.planner->requests[1].back()).contains("→她: "),
            "the @ is pointed out, on the line and in the facts");
    require(h.sent.empty() && h.writer->requests.empty(), "no reply is forced on the planner");
    co_await tasks.join();
}

Task<> test_planner_rests() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    // The planner model is down (its subscription window is used up, say).
    auto fallback = std::make_shared<test::FakeProvider::State>();
    fallback->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        if (after_tool_result(history)) return std::vector{test::tool_call("stay_silent", R"({"reason":"说完了"})")};
        return std::vector{
            test::tool_call("reply", R"({"target":")" + line_number(history, "小明(") + R"(","style":"short","intent":"在"})")};
    };
    h.responder->planner_fallbacks.push_back({.model = "fallback", .provider = test::fake_provider(fallback)});
    h.writer->answer = [](const auto &, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        return std::vector{test::text_answer("在")};
    };
    std::ignore = h.say(1, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" 在吗")});
    co_await h.responder->run_turn(k_chat);
    require(h.sent == std::vector<std::string>{"在"}, "the fallback answered");
    require(h.planner->requests.size() == 2 && h.responder->planner_down_until > h.clock, "two failures in a row: it rests");
    std::ignore = h.say(2, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" 好")});
    co_await h.responder->run_turn(k_chat);
    require(h.planner->requests.size() == 2 && fallback->requests.size() == 4, "while it rests, the fallback is asked directly");
    h.clock = h.responder->planner_down_until;
    std::ignore = h.say(3, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" ？")});
    co_await h.responder->run_turn(k_chat);
    require(h.planner->requests.size() == 3, "after the rest it is tried again");
    co_await tasks.join();
}

Task<> test_conversations_go_on() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    h.planner->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        if (after_tool_result(history)) return analysed("答完了，等他接话", test::tool_call("stay_silent", R"({"reason":"说完了"})"));
        const auto target = line_number(history, "小明(");
        return analysed("小明在问她内存，接着聊",
                        test::tool_call("reply", R"({"target":")" + target + R"(","style":"short","intent":"答一句"})"));
    };
    h.writer->answer = [](const auto &, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        return std::vector{test::text_answer("不好说")};
    };
    require(h.say(1, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" 内存会跌吗")}).has_value(), "asked");
    co_await h.responder->run_turn(k_chat);
    require(h.sent == std::vector<std::string>{"不好说"}, "answered");

    // Seven minutes later, no @, no name: every batch reaches the planner, and
    // it sees what it thought and did before.
    h.clock += 420;
    require(h.say(2, 20001, "小明", {onebot::text_segment("那显卡呢")}).has_value(), "a plain follow-up gets a cycle");
    const auto before = h.planner->requests.size();
    co_await h.responder->run_turn(k_chat);
    const auto &request = h.planner->requests[before];
    std::string seen;
    for (const auto &item : request) seen += text_of(item) + "\n";
    require(seen.contains("小明在问她内存，接着聊") && seen.contains("答完了，等他接话") && seen.contains("她回复了 #"),
            "its earlier analyses and actions are in its history:\n" + seen);
    require(seen.contains("莉莉娅在这里的名字") || seen.contains("在这里的名字"), "the place leads the history");
    require(h.sent.size() == 2, "and it answers the follow-up");
    co_await tasks.join();
}

/// Every text of a request before its final part (the per-round tail): what
/// a later request should start with if the cache is to hold.
std::string stable_text(const provider::History &history) {
    std::string out;
    for (usize i = 0; i < history.size(); ++i) {
        const auto &parts = history[i].parts;
        const usize count = i + 1 == history.size() ? parts.size() - 1 : parts.size();
        for (usize k = 0; k < count; ++k) {
            if (const auto *text = std::get_if<provider::TextPart>(&parts[k])) out += text->text;
            if (const auto *outcome = std::get_if<ToolOutcome>(&parts[k])) out += outcome->receipt;
            if (const auto *call = std::get_if<provider::ToolCall>(&parts[k])) out += call->name;
        }
    }
    return out;
}

/// The cached text part of a writer request.
std::string cached_part(const provider::History &history) {
    for (const auto &item : history) {
        for (const auto &part : item.parts) {
            if (const auto *text = std::get_if<provider::TextPart>(&part); text && text->cache) return text->text;
        }
    }
    return {};
}

Task<> test_cacheable_prompts() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    h.responder->must_speak = true; // so a round is refused and retried
    h.planner->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        const bool refused = after_tool_result(history) && text_of(history.back()).contains("用 reply");
        if (after_tool_result(history) && !refused) return std::vector{test::tool_call("stay_silent", R"({"reason":"说完了"})")};
        if (!refused) return std::vector{test::tool_call("stay_silent", R"({"reason":"不想说"})")};
        return std::vector{
            test::tool_call("reply", R"({"target":")" + line_number(history, "新番") + R"(","style":"short","intent":"接一句"})")};
    };
    h.writer->answer = [](const auto &, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        return std::vector{test::text_answer("确实")};
    };
    std::ignore = h.say(1, 20001, "小明", {onebot::text_segment("新番好看吗")});
    co_await h.responder->run_turn(k_chat);
    require(
        h.sent == std::vector<std::string>{"确实"}, "the refused silence became a reply; sent: " + [&] {
            std::string all;
            for (const auto &text : h.sent) all += "[" + text + "]";
            return all;
        }());
    std::ignore = h.say(2, 20002, "小红", {onebot::text_segment("还行")});
    co_await h.responder->run_turn(k_chat);

    const auto &requests = h.planner->requests;
    require(requests.size() >= 4, "two turns of rounds");
    for (const auto &tools : h.planner->tools) {
        require(tools.size() == h.planner->tools.front().size(), "the same tools every round");
    }
    for (usize i = 1; i < requests.size(); ++i) {
        const auto earlier = stable_text(requests[i - 1]);
        require(stable_text(requests[i]).starts_with(earlier), "each request extends the one before: the cached prefix holds");
        require(!earlier.contains("现在是"), "the time is never in the stable part");
    }
    const auto writer_log = cached_part(h.writer->requests.front());
    require(writer_log == "小明: 新番好看吗", "the writer's log part is cached too: " + writer_log);
    co_await tasks.join();
}

Task<> test_the_planner_searches() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    h.config.model.web_search = true;
    h.planner->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        std::string seen;
        for (const auto &item : history) seen += text_of(item);
        if (seen.contains("她回复了")) return analysed("答完了", test::tool_call("stay_silent", R"({"reason":"说完了"})"));
        if (seen.contains("查到的：")) {
            const auto target = line_number(history, "西湖区");
            return analysed("查到了，告诉他",
                            test::tool_call("reply", R"({"target":")" + target + R"(","style":"short","intent":"说有小雨"})"));
        }
        return analysed("她不知道，先查", test::tool_call("search_web", R"({"query":"杭州西湖区今天下雨吗"})"));
    };
    // One fake answers both the search call (offered only the hosted search
    // tool) and the writer.
    h.writer->answer = [](const auto &,
                          const std::vector<provider::ToolDefinition> &tools) -> std::optional<std::vector<provider::OutputItem>> {
        if (!tools.empty() && tools.front().kind == provider::ToolKind::WEB_SEARCH)
            return std::vector{test::text_answer("西湖区今天下午有小雨")};
        return std::vector{test::text_answer("下午有点小雨")};
    };
    std::ignore = h.say(1, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" 西湖区今天下雨吗")});
    co_await h.responder->run_turn(k_chat);
    require(std::ranges::contains(h.planner->tools.front(), std::string("search_web"), &provider::ToolDefinition::name),
            "the planner can search the web");
    std::string second;
    for (const auto &item : h.planner->requests[1]) second += text_of(item);
    require(second.contains("查到的：西湖区今天下午有小雨"), "it sees what it found before deciding");
    require(h.sent == std::vector<std::string>{"下午有点小雨"}, "then replies");
    const auto &writer = h.writer->requests.back();
    require(text_of(writer.back()).contains("你刚去查过了，情况如下：西湖区今天下午有小雨"), "the writer knows what was found");
    co_await tasks.join();
}

Task<> test_pokes_and_repeats() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    h.planner->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        std::string seen;
        for (const auto &item : history) seen += text_of(item);
        if (after_tool_result(history)) return analysed("好了", test::tool_call("stay_silent", R"({"reason":"好了"})"));
        if (seen.contains("有人戳了戳她")) {
            return analysed("小明戳她，戳回去", test::tool_call("poke", R"({"target":")" + line_number(history, "戳了戳") + R"("})"));
        }
        return analysed("在复读，不跟", test::tool_call("stay_silent", R"({"reason":"不跟"})"));
    };
    // People repeating: a fact for the planner, nothing is sent on its own.
    std::ignore = h.say(1, 20001, "小明", {onebot::text_segment("6")});
    std::ignore = h.say(2, 20002, "小红", {onebot::text_segment("6")});
    co_await h.responder->run_turn(k_chat);
    require(text_of(h.planner->requests.back().back()).contains("最后 2 个人在复读「6」") && h.sent.empty(),
            "a 复读 chain is pointed out; she only joins if the planner says so");

    // Poked: the planner looks at once and decides to poke back.
    h.responder->notice(
        {.time = h.clock, .notice_type = "notify", .sub_type = "poke", .group_id = k_group, .user_id = 20001, .target_id = k_self});
    auto &flow = h.responder->chats[k_chat].flow;
    require(flow.addressed && flow.unseen.size() == 1, "a poke on her makes the planner look");
    co_await h.responder->run_turn(k_chat);
    std::string last;
    for (const auto &item : h.planner->requests.back()) last += text_of(item);
    require(last.contains("没戳成"), "it tried to poke back (no live session here): " + last);
    co_await tasks.join();
}

Task<> test_stickers() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    const auto dir = std::filesystem::temp_directory_path() /
                     ("lilia-test-stickers-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir / "stickers");
    std::ofstream(dir / "stickers" / "a.png", std::ios::binary) << "PNG";
    h.config.stickers.enabled = true;
    sticker::Collector collector(h.config.stickers, dir, h.store, [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    h.responder->stickers = &collector;
    const auto add = [&](std::string sha1, std::string description, std::string tags) {
        const auto id = *h.store.add_sticker({.sha1 = sha1, .path = "stickers/a.png", .media_type = "image/png", .first_seen = 1});
        std::ignore = h.store.describe_sticker(id, description, tags, "registered");
        return id;
    };
    std::ignore = add("h1", "一只猫开心地转圈", "开心、好耶");
    const auto speechless = add("h2", "一只猫瞪大眼睛，配字\"啊？\"", "无语、震惊");

    // The planner looks for a sticker, sees the candidates and sends one.
    h.planner->answer = [speechless](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        const auto last = text_of(history.back());
        if (last.contains("她发了表情包")) return analysed("发完了", test::tool_call("stay_silent", R"({"reason":"发完了"})"));
        if (last.contains("找到这些")) {
            return analysed("这张瞪眼的合适", test::tool_call("send_sticker", std::format(R"({{"sticker":"s{}"}})", speechless)));
        }
        return analysed("有点无语，发个表情", test::tool_call("find_sticker", R"({"feeling":"无语"})"));
    };
    std::ignore = h.say(1, 20001, "小明", {onebot::text_segment("我把作业落在地铁上了")});
    co_await h.responder->run_turn(k_chat);
    const auto &requests = h.planner->requests;
    require(requests.size() == 3 && text_of(requests[1].back()).starts_with(std::format("找到这些：\ns{} 标签：无语、震惊", speechless)),
            "find_sticker shows the closest stickers first: " + text_of(requests[1].back()));
    require(h.outgoing.size() == 1 && h.outgoing[0].front().type == "image" &&
                h.stored_text(h.next_sent) == "[表情: 一只猫瞪大眼睛，配字\"啊？\"]",
            "the sticker goes out and is stored as what it shows");
    require(text_of(requests[0].front()).contains("先用 find_sticker"), "the planner is told how to use stickers");

    // 复读 of a sticker sends the same sticker again.
    onebot::Segment cat{.type = "image"};
    cat.data["url"] = "https://multimedia.nt.qq.com.cn/cat";
    cat.data["sub_type"] = i64{1};
    cat.data["summary"] = "[动画表情]";
    h.clock += 120;
    std::ignore = h.say(2, 20002, "小红", {cat});
    h.planner->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        if (after_tool_result(history)) return analysed("好了", test::tool_call("stay_silent", R"({"reason":"好了"})"));
        return analysed("跟一张", test::tool_call("repeat", R"({"target":")" + line_number(history, "小红") + R"("})"));
    };
    co_await h.responder->run_turn(k_chat);
    require(h.outgoing.size() == 2 && h.outgoing[1].front().type == "image" &&
                onebot::data_string(h.outgoing[1].front(), "file") == "https://multimedia.nt.qq.com.cn/cat",
            "a sticker is repeated as the same sticker");
    co_await tasks.join();
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

Task<> test_several_calls_a_round() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    h.writer->answer = [](const auto &, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        return std::vector{test::text_answer("在追")};
    };
    // A reply and the end of the turn in one answer: both run, in order.
    h.planner->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        const auto target = line_number(history, "新番");
        return std::vector{test::text_answer("小明在问她，答一句就够了"),
                           test::tool_call("reply", R"({"target":")" + target + R"(","style":"short","intent":"说在追"})", "toolu_1"),
                           test::tool_call("stay_silent", R"({"reason":"说完了"})", "toolu_2")};
    };
    std::ignore = h.say(1, 20001, "小明", {onebot::at_segment(k_self), onebot::text_segment(" 新番追了吗")});
    co_await h.responder->run_turn(k_chat);
    require(h.sent == std::vector<std::string>{"在追"} && h.planner->requests.size() == 1, "one round: she replied, then stopped");
    const auto &record = h.responder->chats[k_chat].records.back();
    require(record.outcomes.size() == 2 && record.outcomes[0].receipt.starts_with("她回复了") && record.outcomes[1].receipt == "好。",
            "each call has its result");

    // Calls after a wait are not run: the planner looks again first.
    h.planner->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        if (after_tool_result(history)) return std::vector{test::tool_call("stay_silent", R"({"reason":"算了"})")};
        const auto target = line_number(history, "那你");
        return std::vector{test::tool_call("wait", R"({"seconds":5})", "toolu_3"),
                           test::tool_call("reply", R"({"target":")" + target + R"(","style":"short","intent":"再答"})", "toolu_4")};
    };
    std::ignore = h.say(2, 20001, "小明", {onebot::text_segment("那你看第几集了")});
    co_await h.responder->run_turn(k_chat);
    require(h.sent.size() == 1, "the reply after the wait was not sent");
    const auto &second = h.planner->requests.back();
    require(text_of(second.back()).contains("等了") && text_of(second.back()).contains("没有执行"), "the planner is told which ran");
    co_await tasks.join();
}

Task<> test_private_chats() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    h.config.chat.private_users = {20001};
    const store::ChatKey friend_chat{.type = "private", .id = 20001};
    const auto say = [&](i64 id, std::string text) {
        h.clock += 5;
        return h.responder->ingest({.time = h.clock,
                                    .self_id = k_self,
                                    .message_type = "private",
                                    .message_id = id,
                                    .user_id = 20001,
                                    .sender = {.user_id = 20001, .nickname = "小明"},
                                    .message = {onebot::text_segment(std::move(text))}});
    };
    // The planner waits; the friend writes again, which ends the wait.
    h.responder->realtime = true;
    h.responder->observer = [&](const store::ChatKey &, const chat::Plan &plan, const auto &) {
        if (plan.wait > 0) {
            tasks.spawn([](auto *say) -> Task<> {
                std::ignore = (*say)(2, "我是说今天");
                co_return;
            }(&say));
        }
    };
    h.planner->answer = [](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        if (text_of(history.back()).contains("等了")) return std::vector{test::tool_call("stay_silent", R"({"reason":"等他说完"})")};
        return std::vector{test::tool_call("wait", R"({"seconds":60})")};
    };
    require(say(1, "你说").has_value(), "a private message is answered");
    const auto started = std::chrono::steady_clock::now();
    co_await h.responder->run_turn(friend_chat);
    require(std::chrono::steady_clock::now() - started < std::chrono::seconds(30), "the new message ended the wait");
    const auto &requests = h.planner->requests;
    require(requests.size() == 2 && text_of(requests[1].back()).contains("对方发来了新消息") &&
                text_of(requests[1].back()).contains("我是说今天"),
            "and the planner sees it");
    const auto system = text_of(requests[0].front());
    require(system.contains("这是私聊") && !system.contains("先弄清每条新消息是对谁说的"), "the private rules, not the group's: " + system);
    co_await tasks.join();
}

Task<> test_history_survives_many_turns() {
    lighter::TaskGroup<> tasks(lighter::EventLoop::current());
    Harness h(tasks);
    h.planner->answer = [](const provider::History &, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        return analysed("只是闲聊，不用接", test::tool_call("stay_silent", R"({"reason":"闲聊"})"));
    };
    // A fresh database: row ids start at 1, so the window's anchor is 0.
    for (i64 id = 1; id <= 9; ++id) {
        std::ignore = h.say(id, 20001, "小明", {onebot::text_segment("第" + std::to_string(id) + "句")});
        h.responder->chats[k_chat].flow.backoff_until = 0; // look every time
        co_await h.responder->run_turn(k_chat);
    }
    const auto &requests = h.planner->requests;
    require(requests.size() == 9, "a cycle per line");
    for (usize i = 1; i < requests.size(); ++i) {
        require(stable_text(requests[i]).starts_with(stable_text(requests[i - 1])),
                "nothing is folded while the window has not moved (turn " + std::to_string(i + 1) + ")");
    }
    co_await tasks.join();
}

} // namespace

int main() {
    try {
        for (auto *test : {&test_quotes_and_recalls, &test_mutes_and_notices, &test_the_planner_decides, &test_cacheable_prompts,
                           &test_planner_rests, &test_conversations_go_on, &test_history_survives_many_turns, &test_the_planner_searches,
                           &test_pokes_and_repeats, &test_stickers, &test_several_calls_a_round, &test_private_chats}) {
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
