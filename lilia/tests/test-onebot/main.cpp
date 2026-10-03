#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <glaze/json.hpp>

#include <lighter/async/io/loop.h>
#include <lighter/async/io/stream.h>
#include <lighter/async/runtime/task.h>

#include <lilia/onebot/message.h>
#include <lilia/onebot/session.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia::onebot;
using lilia::test::FakeWsPeer;
using lilia::test::require;
using namespace lighter::types;

constexpr i64 k_self_id = 10001;
constexpr i64 k_group_id = 987654321;

// Shape captured from NapCat 4.18 with messagePostFormat "array".
constexpr std::string_view k_group_message = R"json({
  "self_id": 10001, "user_id": 20002, "time": 1759400000, "message_id": -2147480001, "message_seq": 4321,
  "real_id": 4321, "real_seq": "1234", "message_type": "group",
  "sender": {"user_id": 20002, "nickname": "魔王", "card": "", "role": "member"},
  "raw_message": "[CQ:reply,id=1500][CQ:at,qq=10001] 看看这个", "font": 14, "sub_type": "normal",
  "message": [
    {"type": "reply", "data": {"id": "1500"}},
    {"type": "at", "data": {"qq": "10001", "name": "lilia"}},
    {"type": "text", "data": {"text": " 看看这个"}},
    {"type": "image", "data": {"summary": "[动画表情]", "file": "a.gif", "sub_type": 1, "url": "https://example.invalid/a.gif"}},
    {"type": "mface", "data": {"emoji_package_id": 1, "emoji_id": "x", "key": "k", "summary": "[开心]"}},
    {"type": "image", "data": {"summary": "", "file": "b.jpg", "sub_type": 0}},
    {"type": "poke", "data": {}}
  ],
  "message_format": "array", "post_type": "message", "group_id": 987654321, "group_name": "测试群"
})json";

constexpr std::string_view k_heartbeat = R"json({"time": 1759400001, "self_id": 10001, "post_type": "meta_event",
  "meta_event_type": "heartbeat", "status": {"online": true, "good": true}, "interval": 30000})json";

void test_decode_and_render() {
    auto event = decode_event(k_group_message, "message");
    require(event.has_value(), event ? std::string() : std::string(event.error().message()));
    const auto *message = std::get_if<MessageEvent>(&*event);
    require(message != nullptr, "post_type message must decode to MessageEvent");
    require(message->is_group() && *message->group_id == k_group_id, "group id");
    require(message->message_id == -2147480001, "negative message ids must survive");
    require(message->sender.nickname == "魔王", "sender nickname");
    require(mentions(message->message, k_self_id), "the bot is mentioned");
    require(!mentions(message->message, 30003), "others are not mentioned");
    require(reply_target(message->message) == 1500, "reply target parsed from a string id");
    require(text_only(message->message) == " 看看这个", "text_only keeps only text segments");

    const auto rendered = render_plain(message->message);
    require(rendered == "@lilia 看看这个[表情: 动画表情][表情: 开心][图片][poke]", "render_plain: " + rendered);

    auto heartbeat = decode_event(k_heartbeat, "meta_event");
    require(heartbeat.has_value(), "heartbeat must decode");
    const auto *meta = std::get_if<MetaEvent>(&*heartbeat);
    require(meta && meta->meta_event_type == "heartbeat" && meta->interval == 30000, "heartbeat fields");

    auto notice = decode_event(R"({"post_type":"notice","notice_type":"group_increase"})", "notice");
    require(notice.has_value() && std::holds_alternative<OtherEvent>(*notice), "unknown post types become OtherEvent");

    const Message outgoing{reply_segment(-5), at_segment(k_self_id), text_segment("hi")};
    const auto encoded = glz::write_json(outgoing).value_or("");
    require(encoded == R"([{"type":"reply","data":{"id":"-5"}},{"type":"at","data":{"qq":"10001"}},{"type":"text","data":{"text":"hi"}}])",
            "outgoing segments encode as OneBot arrays: " + encoded);
}

struct Request {
    std::string action;
    glz::generic_i64 params;
    std::string echo;
};

Request parse_request(std::string_view text) {
    Request request;
    auto ec = glz::read<glz::opts{.error_on_unknown_keys = false}>(request, text);
    require(!ec, "client request must be valid JSON: " + std::string(text));
    return request;
}

std::string response(std::string_view echo, std::string_view data) {
    return std::string(R"({"status":"ok","retcode":0,"data":)") + std::string(data) + R"(,"message":"","wording":"","echo":")" +
           std::string(echo) + R"("})";
}

lighter::Task<> fake_napcat(lighter::Tcp::Acceptor listener) {
    auto owned = co_await FakeWsPeer::accept(listener);
    auto &peer = *owned;
    listener = {};
    require(peer.has_header("Authorization: Bearer token-123"), "access token must be sent as a bearer header");

    auto login = parse_request(co_await peer.read_text());
    require(login.action == "get_login_info", "first action");
    // An event may arrive between a request and its response.
    co_await peer.send_text(k_group_message);
    co_await peer.send_text(response(login.echo, R"({"user_id":10001,"nickname":"lilia"})"));

    auto send = parse_request(co_await peer.read_text());
    require(send.action == "send_group_msg", "second action");
    require(send.params["group_id"].get<i64>() == k_group_id, "group_id param");
    require(send.params["message"].get_array().size() == 2, "message param segments");
    co_await peer.send_text(response(send.echo, R"({"message_id":555})"));

    auto unsupported = parse_request(co_await peer.read_text());
    co_await peer.send_text(R"({"status":"failed","retcode":1404,"data":null,"message":"不支持的Api","wording":"不支持的Api","echo":")" +
                            unsupported.echo + R"("})");

    // Never answered: the close must fail it.
    std::ignore = parse_request(co_await peer.read_text());
    co_await peer.send_close(1001);
    std::ignore = co_await peer.read_frame(); // the client's close echo
}

struct Observed {
    std::vector<Event> events;
    std::optional<OneBotError> run_end;
    bool done = false;
};

lighter::Task<> run_session(std::shared_ptr<Session> session, Observed &observed) {
    auto ended = co_await session->run([&observed](Event event) { observed.events.push_back(std::move(event)); });
    require(ended.has_error(), "run() always ends with an error");
    observed.run_end = std::move(ended).error();
}

lighter::Task<> client(i32 port, Observed &observed) {
    auto opened = co_await Session::open({
        .connect = {.host = "127.0.0.1", .port = port},
        .access_token = "token-123",
    });
    require(opened.has_value(), opened ? std::string() : opened.error().message());
    auto session = std::move(*opened);
    lighter::EventLoop::current().schedule(run_session(session, observed));

    auto login = co_await session->get_login_info();
    require(login.has_value() && login->user_id == k_self_id && login->nickname == "lilia", "get_login_info result");
    require(observed.events.size() == 1 && std::holds_alternative<MessageEvent>(observed.events[0]),
            "the interleaved event must be delivered before the response resolves");

    auto sent = co_await session->send_group_msg(k_group_id, {reply_segment(1500), text_segment("你好")});
    require(sent.has_value() && sent->message_id == 555, "send_group_msg result");

    auto failed = co_await session->call("unsupported_action", "{}");
    require(failed.has_error() && failed.error().kind == OneBotErrorKind::ACTION_FAILED && failed.error().retcode == 1404,
            "a failed status must surface as ACTION_FAILED with its retcode");

    auto orphaned = co_await session->call("get_status", "{}");
    require(orphaned.has_error() && orphaned.error().kind == OneBotErrorKind::DISCONNECTED,
            "a pending call must fail with DISCONNECTED when the connection closes");
    require(!session->is_open(), "session must be closed");
    observed.done = true;
}

void test_session() {
    lighter::EventLoop loop;
    auto listener = lighter::Tcp::listen("127.0.0.1", 0, {}, loop);
    require(static_cast<bool>(listener), "listener");
    auto port = lighter::Tcp::local_port(*listener);
    require(static_cast<bool>(port), "listener port");

    Observed observed;
    auto server_task = fake_napcat(std::move(*listener));
    auto client_task = client(*port, observed);
    loop.schedule(server_task);
    loop.schedule(client_task);
    loop.run();

    std::ignore = server_task.value();
    std::ignore = client_task.value();
    require(observed.done, "client finished");
    require(observed.run_end && observed.run_end->kind == OneBotErrorKind::TRANSPORT && observed.run_end->ws &&
                observed.run_end->ws->close_code == 1001,
            "run() must end with the peer's close");
}

} // namespace

int main() {
    try {
        test_decode_and_render();
        test_session();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
