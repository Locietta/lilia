#include "responder.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <utility>
#include <variant>

#include <glaze/json.hpp>

#include <lighter/async/io/watcher.h>

#include <xmake/version/lilia.hpp>

#include <lilia/chat/context.h>
#include <lilia/chat/trigger.h>
#include <lilia/log.h>
#include <lilia/onebot/message.h>

namespace lilia::chat {

using lighter::Task;

const std::string_view k_default_persona = R"(你是 lilia（中文名莉莉娅），住在朋友们自己搭的服务器 cachy 上，是这个 QQ 群的一员。
- 群里是几个一起玩游戏（Minecraft、饥荒等）和闲聊的朋友，你和大家是平等的朋友关系，不是客服。
- 性格好奇、温和、带一点俏皮。说话简短自然，用口语，偶尔开个玩笑，但不刻意卖萌。
- 别人问问题时认真回答；只是闲聊时就简单接话，不要长篇大论，也不要列清单。
- 不确定的事情直接说不确定，不编造。
- 不要反复强调自己是 AI 或助手。)";

namespace {

store::ChatKey chat_key(const onebot::MessageEvent &event) {
    if (event.is_group()) return {.type = "group", .id = *event.group_id};
    return {.type = "private", .id = event.user_id};
}

store::StoredMessage to_stored(const onebot::MessageEvent &event) {
    return {
        .chat = chat_key(event),
        .message_id = event.message_id,
        .sender_id = event.user_id,
        .sender_name = !event.sender.card.empty() ? event.sender.card : event.sender.nickname,
        .time = event.time,
        .segments_json = glz::write_json(event.message).value_or("[]"),
        .text = onebot::render_plain(event.message),
    };
}

std::string describe(const store::ChatKey &chat) { return chat.type + ":" + std::to_string(chat.id); }

Task<> send_reply(Responder *r, std::shared_ptr<onebot::Session> session, store::ChatKey chat, std::optional<i64> quote, std::string text) {
    onebot::Message message;
    if (quote && chat.type == "group") message.push_back(onebot::reply_segment(*quote));
    message.push_back(onebot::text_segment(text));

    auto sent = co_await (chat.type == "group" ? session->send_group_msg(chat.id, message) : session->send_private_msg(chat.id, message));
    if (sent.has_error()) {
        log::error("send to {} failed: {}", describe(chat), sent.error().message());
        co_return;
    }
    auto stored = r->store.add_message({
        .chat = chat,
        .message_id = sent->message_id,
        .sender_id = r->self.user_id,
        .sender_name = r->self.nickname,
        .time = r->now(),
        .segments_json = glz::write_json(message).value_or("[]"),
        .text = onebot::render_plain(message),
        .from_self = true,
    });
    if (stored.has_error()) log::error("cannot record sent message: {}", stored.error().message());
}

Task<> respond(Responder *r, store::ChatKey chat, std::vector<store::StoredMessage> triggers) {
    auto session = r->session;
    if (!session || !session->is_open()) {
        log::warn("no live OneBot session; dropping {} trigger(s) in {}", triggers.size(), describe(chat));
        co_return;
    }
    const auto &settings = r->config.chat;
    const i64 now = r->now();
    const i64 day = local_day_start(now, settings.utc_offset_minutes);

    auto used = r->store.tokens_since(day);
    if (used.has_error()) log::error("cannot read token usage: {}", used.error().message());
    if (used.has_value() && *used >= settings.daily_token_budget) {
        auto &state = r->chats[chat];
        log::warn("daily token budget exhausted ({} >= {})", *used, settings.daily_token_budget);
        if (state.budget_notice_day != day) {
            state.budget_notice_day = day;
            co_await send_reply(r, session, chat, triggers.back().message_id, "今天的额度已经用完了，明天再来找我吧。");
        }
        co_return;
    }

    auto recent = r->store.recent_messages(chat, settings.context_messages);
    if (recent.has_error()) log::error("cannot read chat log: {}", recent.error().message());

    ContextRequest request{
        .persona = r->persona,
        .info = {.chat = chat, .chat_name = r->chats[chat].name, .self_id = r->self.user_id, .self_name = r->self.nickname},
        .recent = recent.has_value() ? *std::move(recent) : std::vector<store::StoredMessage>{},
        .triggers = triggers,
        .now = now,
        .utc_offset_minutes = settings.utc_offset_minutes,
        .log_bytes = settings.context_bytes,
    };
    const auto history = build_context(request);

    std::string answer;
    const provider::StreamCallbacks callbacks{
        .on_item_completed =
            [&answer](const provider::OutputItem &item) {
                if (const auto *message = std::get_if<provider::AssistantMessageItem>(&item)) {
                    for (const auto &part : message->parts) answer += part.text;
                }
            },
    };
    auto completion = co_await r->provider->complete(history, {}, callbacks);

    store::LlmCall call{.time = now, .chat = chat, .model = r->config.model.model};
    if (completion.has_error()) {
        call.error = completion.error().message();
        call.request_id = completion.error().request_id;
        log::error("model call for {} failed: {}", describe(chat), call.error);
    } else {
        call.model = completion->model.empty() ? call.model : completion->model;
        call.input_tokens = completion->usage.input_tokens;
        call.output_tokens = completion->usage.output_tokens;
        call.cache_read_tokens = completion->usage.cache_read_tokens;
        call.cache_write_tokens = completion->usage.cache_write_tokens;
        call.stop = completion->stop_detail;
        call.request_id = completion->request_id;
        log::info("model call for {}: {} in / {} out / {} cached tokens, stop {}", describe(chat), call.input_tokens, call.output_tokens,
                  call.cache_read_tokens, call.stop);
    }
    if (auto recorded = r->store.add_llm_call(call); recorded.has_error()) {
        log::error("cannot record model call: {}", recorded.error().message());
    }
    if (completion.has_error()) co_return;

    auto text = clean_reply(answer, request.info, settings.aliases, settings.max_reply_bytes);
    if (text.empty()) {
        log::warn("model returned an empty reply for {}", describe(chat));
        co_return;
    }
    co_await send_reply(r, std::move(session), chat, triggers.back().message_id, std::move(text));
}

Task<> reply_worker(Responder *r, store::ChatKey chat) {
    // Leave the reader loop's call stack before doing any work.
    co_await lighter::yield();
    for (;;) {
        auto &state = r->chats[chat];
        if (state.pending.empty()) {
            state.busy = false;
            co_return;
        }
        const i64 wait = state.last_reply + r->config.chat.min_reply_interval_seconds - r->now();
        if (wait > 0) co_await lighter::sleep(std::chrono::seconds(wait));

        auto triggers = std::exchange(r->chats[chat].pending, {});
        co_await respond(r, chat, std::move(triggers));
        r->chats[chat].last_reply = r->now();
    }
}

Task<> handle_admin(Responder *r, store::StoredMessage command, std::string arguments) {
    auto session = r->session;
    if (!session) co_return;
    std::string reply;
    if (arguments == "status") {
        const i64 day = local_day_start(r->now(), r->config.chat.utc_offset_minutes);
        auto used = r->store.tokens_since(day);
        reply = std::format("lilia {} ({}{})\n模型：{}\n今日 token：{} / {}\n已允许群：{} 个", xmake::version::version,
                            xmake::version::commit.substr(0, 8), xmake::version::dirty ? "-dirty" : "", r->config.model.model,
                            used.has_value() ? *used : 0, r->config.chat.daily_token_budget, r->config.chat.groups.size());
    } else if (arguments == "reload") {
        auto reloaded = r->reload_persona();
        reply = reloaded.has_value() ? "人设已重新加载。" : "重新加载失败：" + reloaded.error().message();
    } else {
        reply = "用法：/lilia status | /lilia reload";
    }
    co_await send_reply(r, std::move(session), command.chat, command.message_id, std::move(reply));
}

} // namespace

Responder::Responder(const Config &config, store::Store &store, provider::Provider provider, std::filesystem::path persona_path,
                     Spawn spawn)
    : now([] { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }),
      config(config), store(store), provider(std::move(provider)), persona_path(std::move(persona_path)), persona(k_default_persona),
      spawn(std::move(spawn)) {}

void Responder::attach(std::shared_ptr<onebot::Session> live, onebot::LoginInfo login) {
    session = std::move(live);
    self = std::move(login);
}

void Responder::detach() noexcept { session.reset(); }

Result<void> Responder::reload_persona() {
    std::ifstream file(persona_path, std::ios::binary);
    if (!file) {
        persona = std::string(k_default_persona);
        log::info("no persona file at {}; using the built-in persona", persona_path.string());
        return {};
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    auto text = buffer.str();
    if (text.find_first_not_of(" \t\r\n") == std::string::npos) {
        return lighter::outcome_error(Error::config(persona_path.string() + " is empty"));
    }
    persona = std::move(text);
    log::info("loaded persona from {} ({} bytes)", persona_path.string(), persona.size());
    return {};
}

void Responder::on_event(onebot::Event event) {
    if (const auto *meta = std::get_if<onebot::MetaEvent>(&event)) {
        if (meta->meta_event_type == "lifecycle") log::info("onebot lifecycle: {}", meta->sub_type);
        return;
    }
    // Until get_login_info answers, lilia does not know its own id.
    auto *message = std::get_if<onebot::MessageEvent>(&event);
    if (!message || !session || message->user_id == self.user_id || !chat_allowed(*message, config.chat)) return;

    auto stored = to_stored(*message);
    if (auto added = store.add_message(stored); added.has_error()) log::error("cannot store message: {}", added.error().message());
    auto &state = chats[stored.chat];
    if (message->is_group()) {
        if (message->group_name && !message->group_name->empty()) state.name = *message->group_name;
    } else {
        state.name = stored.sender_name;
    }

    std::string_view arguments;
    if (std::ranges::contains(config.chat.admins, message->user_id) &&
        parse_admin_command(onebot::text_only(message->message), arguments)) {
        spawn(handle_admin(this, std::move(stored), std::string(arguments)));
        return;
    }

    bool replies_to_self = false;
    if (const auto target = onebot::reply_target(message->message)) {
        auto own = store.is_own_message(stored.chat, *target);
        replies_to_self = own.has_value() && *own;
    }
    const auto reason = classify(*message, self.user_id, replies_to_self, config.chat);
    if (reason == TriggerReason::NONE) return;

    log::info("{} trigger in {} from {}", to_string(reason), describe(stored.chat), stored.sender_id);
    const auto chat = stored.chat;
    state.pending.push_back(std::move(stored));
    if (!state.busy) {
        state.busy = true;
        spawn(reply_worker(this, chat));
    }
}

} // namespace lilia::chat
