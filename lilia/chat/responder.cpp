#include "responder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <span>
#include <sstream>
#include <utility>
#include <variant>

#include <glaze/json.hpp>

#include <lighter/async/io/watcher.h>

#include <xmake/bin2obj/persona_md.hpp>
#include <xmake/bin2obj/planner_md.hpp>
#include <xmake/bin2obj/writer_md.hpp>
#include <xmake/version/lilia.hpp>

#include <lilia/chat/delivery.h>
#include <lilia/chat/trigger.h>
#include <lilia/log.h>
#include <lilia/onebot/message.h>

namespace lilia::chat {

using lighter::Task;

namespace {

std::string_view embedded(std::span<const std::byte> content) { return {reinterpret_cast<const char *>(content.data()), content.size()}; }

/// One-off styles that occasionally replace the planner's choice, so lilia
/// does not always sound the same.
constexpr std::array<std::string_view, 4> k_alt_styles{
    "这次只用一两个字或者一个词回应。",
    "这次随口接一句就行，像顺手回的消息。",
    "这次可以带点吐槽的语气。",
    "这次用反问或者疑问的口气回应。",
};

constexpr i64 k_max_debounce_seconds = 15;

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

std::string_view style_name(ReplyStyle style) {
    switch (style) {
        case ReplyStyle::SHORT: return "short";
        case ReplyStyle::NORMAL: return "normal";
        case ReplyStyle::LONG: return "long";
    }
    return "?";
}

Task<std::optional<i64>> send_message(Responder *r, const store::ChatKey &chat, onebot::Message message) {
    if (r->outbox) co_return co_await r->outbox(chat, std::move(message));
    auto session = r->session;
    if (!session || !session->is_open()) {
        log::warn("no live OneBot session; cannot send to {}", describe(chat));
        co_return std::nullopt;
    }
    auto sent = co_await (chat.type == "group" ? session->send_group_msg(chat.id, std::move(message)) :
                                                 session->send_private_msg(chat.id, std::move(message)));
    if (sent.has_error()) {
        log::error("send to {} failed: {}", describe(chat), sent.error().message());
        co_return std::nullopt;
    }
    co_return sent->message_id;
}

/// Sends one message and records it as lilia's own.
Task<bool> send_and_record(Responder *r, const store::ChatKey &chat, onebot::Message message) {
    auto segments_json = glz::write_json(message).value_or("[]");
    auto text = onebot::render_plain(message);
    const auto id = co_await send_message(r, chat, std::move(message));
    if (!id) co_return false;
    auto stored = r->store.add_message({
        .chat = chat,
        .message_id = *id,
        .sender_id = r->self.user_id,
        .sender_name = r->self.nickname,
        .time = r->now(),
        .segments_json = std::move(segments_json),
        .text = std::move(text),
        .from_self = true,
    });
    if (stored.has_error()) log::error("cannot record sent message: {}", stored.error().message());
    co_return true;
}

/// Runs one model call, records its usage, and returns the completed output
/// items (nullopt on failure).
Task<std::optional<std::vector<provider::OutputItem>>> call_model(Responder *r, provider::Provider &provider, std::string model,
                                                                  std::string_view purpose, const store::ChatKey &chat,
                                                                  const provider::History &history,
                                                                  const std::vector<provider::ToolDefinition> &tools) {
    std::vector<provider::OutputItem> items;
    const provider::StreamCallbacks callbacks{
        .on_item_completed = [&items](const provider::OutputItem &item) { items.push_back(item); },
    };
    const i64 started = r->now();
    auto completion = co_await provider->complete(history, tools, callbacks);

    store::LlmCall call{.time = started, .chat = chat, .model = std::move(model)};
    if (completion.has_error()) {
        call.error = completion.error().message();
        call.request_id = completion.error().request_id;
        log::error("{} call for {} failed: {}", purpose, describe(chat), call.error);
    } else {
        if (!completion->model.empty()) call.model = completion->model;
        call.input_tokens = completion->usage.input_tokens;
        call.output_tokens = completion->usage.output_tokens;
        call.cache_read_tokens = completion->usage.cache_read_tokens;
        call.cache_write_tokens = completion->usage.cache_write_tokens;
        call.stop = completion->stop_detail;
        call.request_id = completion->request_id;
        log::info("{} call for {}: {} in / {} out / {} cached tokens", purpose, describe(chat), call.input_tokens, call.output_tokens,
                  call.cache_read_tokens);
    }
    if (auto recorded = r->store.add_llm_call(call); recorded.has_error()) {
        log::error("cannot record model call: {}", recorded.error().message());
    }
    if (completion.has_error()) co_return std::nullopt;
    co_return std::move(items);
}

/// Text after the last non-text item; earlier text is narration such as
/// "let me search for that".
std::string final_text(const std::vector<provider::OutputItem> &items) {
    std::string answer;
    for (const auto &item : items) {
        if (const auto *message = std::get_if<provider::AssistantMessageItem>(&item)) {
            for (const auto &part : message->parts) answer += part.text;
        } else {
            answer.clear();
        }
    }
    return answer;
}

Task<> debounce(Responder *r, const store::ChatKey &chat) {
    const i64 started = r->now();
    for (;;) {
        const auto &flow = r->chats[chat].flow;
        const i64 quiet = flow.addressed ? std::min<i64>(r->config.chat.debounce_seconds, 2) : r->config.chat.debounce_seconds;
        const i64 wait = flow.last_arrival + quiet - r->now();
        if (wait <= 0 || r->now() - started >= k_max_debounce_seconds) co_return;
        co_await lighter::sleep(std::chrono::seconds(wait));
    }
}

Task<> turn_worker(Responder *r, store::ChatKey chat) {
    // Leave the reader loop's call stack before doing any work.
    co_await lighter::yield();
    for (;;) {
        auto &state = r->chats[chat];
        if (!should_plan(state.flow, r->flow_settings(), r->now())) {
            state.busy = false;
            co_return;
        }
        if (r->realtime) co_await debounce(r, chat);
        co_await r->run_turn(chat);
    }
}

Task<> handle_admin(Responder *r, store::StoredMessage command, std::string arguments) {
    std::string reply;
    if (arguments == "status") {
        const i64 day = local_day_start(r->now(), r->config.chat.utc_offset_minutes);
        auto used = r->store.tokens_since(day);
        reply = std::format("lilia {} ({}{})\n模型：{} / 规划 {}\n今日 token：{} / {}\n已允许群：{} 个", xmake::version::version,
                            xmake::version::commit.substr(0, 8), xmake::version::dirty ? "-dirty" : "", r->config.model.model,
                            r->config.model.planner_model, used.has_value() ? *used : 0, r->config.chat.daily_token_budget,
                            r->config.chat.groups.size());
    } else if (arguments == "reload") {
        auto reloaded = r->reload_persona();
        reply = reloaded.has_value() ? "人设已重新加载。" : "重新加载失败：" + reloaded.error().message();
    } else {
        reply = "用法：/lilia status | /lilia reload";
    }
    onebot::Message message;
    if (command.chat.type == "group") message.push_back(onebot::reply_segment(command.message_id));
    message.push_back(onebot::text_segment(std::move(reply)));
    co_await send_and_record(r, command.chat, std::move(message));
}

} // namespace

std::string_view default_persona() { return embedded(xmake::persona_md); }

std::vector<std::string> sample_habits(store::Store &store, usize count, std::mt19937 &rng) {
    auto all = store.expressions(500);
    if (all.has_error() || count == 0) return {};
    // The most learned few, then a random spread of the rest.
    const usize top = std::min<usize>({all->size(), count / 3});
    std::vector<store::Expression> picked(all->begin(), all->begin() + static_cast<std::ptrdiff_t>(top));
    std::vector<store::Expression> rest(all->begin() + static_cast<std::ptrdiff_t>(top), all->end());
    std::ranges::shuffle(rest, rng);
    for (auto &expression : rest) {
        if (picked.size() >= count) break;
        picked.push_back(std::move(expression));
    }
    std::ranges::shuffle(picked, rng);
    std::vector<std::string> habits;
    for (const auto &expression : picked) habits.push_back(std::format("当{}时，可以{}", expression.situation, expression.style));
    return habits;
}

Responder::Responder(const Config &config, store::Store &store, provider::Provider planner, provider::Provider searcher,
                     provider::Provider writer, std::filesystem::path persona_path, Spawn spawn)
    : now([] { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }),
      config(config), store(store), planner(std::move(planner)), searcher(std::move(searcher)), writer(std::move(writer)),
      persona_path(std::move(persona_path)), persona(default_persona()), spawn(std::move(spawn)) {}

void Responder::attach(std::shared_ptr<onebot::Session> live, onebot::LoginInfo login) {
    session = std::move(live);
    self = std::move(login);
}

void Responder::detach() noexcept { session.reset(); }

FlowSettings Responder::flow_settings() const {
    return {.talk_value = config.chat.talk_value, .window_seconds = config.chat.conversation_window_seconds};
}

Result<void> Responder::reload_persona() {
    std::ifstream file(persona_path, std::ios::binary);
    if (!file) {
        persona = std::string(default_persona());
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

std::optional<store::ChatKey> Responder::ingest(const onebot::MessageEvent &message) {
    if (message.user_id == self.user_id || !chat_allowed(message, config.chat)) return std::nullopt;

    auto stored = to_stored(message);
    if (auto added = store.add_message(stored); added.has_error()) log::error("cannot store message: {}", added.error().message());
    if (archiver && session && !archive::collect_media(message.message).empty()) {
        spawn(archiver->archive(session, stored, message.message));
    }
    auto &state = chats[stored.chat];
    if (message.is_group()) {
        if (message.group_name && !message.group_name->empty()) state.name = *message.group_name;
    } else {
        state.name = stored.sender_name;
    }

    std::string_view arguments;
    if (std::ranges::contains(config.chat.admins, message.user_id) && parse_admin_command(onebot::text_only(message.message), arguments)) {
        spawn(handle_admin(this, std::move(stored), std::string(arguments)));
        return std::nullopt;
    }

    bool replies_to_self = false;
    if (const auto target = onebot::reply_target(message.message)) {
        auto own = store.is_own_message(stored.chat, *target);
        replies_to_self = own.has_value() && *own;
    }
    const auto reason = classify(message, self.user_id, replies_to_self, config.chat);
    if (reason != TriggerReason::NONE) {
        log::info("{} from {} in {}", to_string(reason), stored.sender_id, describe(stored.chat));
        state.flow.addressed = stored.message_id;
    }
    const auto chat = stored.chat;
    state.flow.unseen.push_back(std::move(stored));
    state.flow.last_arrival = now();
    if (!should_plan(state.flow, flow_settings(), now())) return std::nullopt;
    return chat;
}

void Responder::on_event(onebot::Event event) {
    if (const auto *meta = std::get_if<onebot::MetaEvent>(&event)) {
        if (meta->meta_event_type == "lifecycle") log::info("onebot lifecycle: {}", meta->sub_type);
        return;
    }
    // Until get_login_info answers, lilia does not know its own id.
    const auto *message = std::get_if<onebot::MessageEvent>(&event);
    if (!message || !session) return;
    // Every group may teach her how people talk; only allowed chats go further.
    if (learner) learner->observe(*message, self.user_id);
    const auto chat = ingest(*message);
    if (!chat) return;
    auto &state = chats[*chat];
    if (state.busy) return; // the running worker looks again after its turn
    state.busy = true;
    spawn(turn_worker(this, *chat));
}

Task<> Responder::run_turn(store::ChatKey chat) {
    auto &state = chats[chat];
    auto fresh = std::exchange(state.flow.unseen, {});
    const auto addressed = std::exchange(state.flow.addressed, std::nullopt);
    if (fresh.empty()) co_return;

    const auto &settings = config.chat;
    const i64 started = now();
    const i64 day = local_day_start(started, settings.utc_offset_minutes);
    auto used = store.tokens_since(day);
    if (used.has_error()) log::error("cannot read token usage: {}", used.error().message());
    if (used.has_value() && *used >= settings.daily_token_budget) {
        log::warn("daily token budget exhausted ({} >= {})", *used, settings.daily_token_budget);
        if (addressed && state.budget_notice_day != day) {
            state.budget_notice_day = day;
            onebot::Message notice{onebot::text_segment("今天的额度用完了，明天再找我吧")};
            co_await send_and_record(this, chat, std::move(notice));
        }
        co_return;
    }

    auto recent = store.recent_messages(chat, settings.context_messages);
    if (recent.has_error()) log::error("cannot read chat log: {}", recent.error().message());
    auto log_lines = recent.has_value() ? *std::move(recent) : fresh;

    const ChatInfo info{.chat = chat, .chat_name = state.name, .name = settings.name, .self_id = self.user_id, .self_name = self.nickname};
    PlannerInput planner_input{
        .prompt = fill_template(embedded(xmake::planner_md), {{"name", settings.name}, {"persona", persona}}),
        .info = info,
        .recent = log_lines,
        .addressed = addressed.has_value(),
        .now = started,
        .utc_offset_minutes = settings.utc_offset_minutes,
        .log_bytes = settings.context_bytes,
    };
    if (config.learning.enabled) planner_input.habits = sample_habits(store, config.learning.candidates, rng);
    for (const auto &message : fresh) planner_input.fresh.push_back(message.message_id);
    const auto request = build_planner_request(planner_input);

    const auto planned = co_await call_model(this, planner, config.model.planner_model, "planner", chat, request.history, request.tools);
    auto plan = planned ? parse_plan(*planned, request.ids) : Plan{.reason = "planner call failed"};
    if (!plan.reply && addressed) {
        // Ignoring someone who called her reads as broken, not as a choice.
        log::info("planner stayed silent although addressed ({}); replying anyway", plan.reason);
        plan = {.reply = true, .target = *addressed};
    }
    if (!plan.reply) {
        log::info("{} stays silent: {}", describe(chat), plan.reason);
        note_silent(state.flow, now());
        if (observer) observer(chat, plan, {});
        co_return;
    }

    auto target = std::ranges::find(log_lines, plan.target, &store::StoredMessage::message_id);
    if (target != log_lines.end() && target->from_self) {
        // She answers others, not herself; fall back to the newest new message.
        plan.target = addressed.value_or(fresh.back().message_id);
        target = std::ranges::find(log_lines, plan.target, &store::StoredMessage::message_id);
    }
    if (target == log_lines.end()) {
        log::warn("planner target {} is not in the chat log", plan.target);
        co_return;
    }
    log::info("{} replies to {} ({}, quote {}, search {}, habits {}): {}", describe(chat), plan.target, style_name(plan.style), plan.quote,
              plan.habits.size(), plan.search, plan.intent);

    std::string style_override;
    if (plan.style != ReplyStyle::LONG && std::bernoulli_distribution(std::clamp(settings.alt_style_probability, 0.0, 1.0))(rng)) {
        style_override = k_alt_styles[std::uniform_int_distribution<usize>(0, k_alt_styles.size() - 1)(rng)];
    }
    std::string reference;
    if (plan.search && config.model.web_search) {
        // A dedicated call: writers asked to search "if needed" often don't.
        const auto search_history = build_search_history(info, *target, plan, started, settings.utc_offset_minutes);
        const std::vector<provider::ToolDefinition> search_tools{{.kind = provider::ToolKind::WEB_SEARCH, .name = "web_search"}};
        if (const auto found =
                co_await call_model(this, searcher, config.model.planner_model, "search", chat, search_history, search_tools)) {
            reference = clean_reply(final_text(*found), info, {}, 2'000);
        }
        log::info("search for {} returned {} bytes", describe(chat), reference.size());
        log::debug("search reference: {}", reference);
    }
    const auto history = build_writer_history({
        .persona = persona,
        .rules = std::string(embedded(xmake::writer_md)),
        .info = info,
        .recent = log_lines,
        .target = *target,
        .plan = plan,
        .style_override = style_override,
        .now = started,
        .utc_offset_minutes = settings.utc_offset_minutes,
        .log_bytes = settings.context_bytes,
        .reference = reference,
        .habits =
            [&] {
                std::vector<std::string> picked;
                for (const auto index : plan.habits) {
                    if (index < planner_input.habits.size()) picked.push_back(planner_input.habits[index]);
                }
                return picked;
            }(),
    });
    const auto written = co_await call_model(this, writer, config.model.model, "writer", chat, history, {});
    if (!written) co_return;
    const auto budget = !style_override.empty() || plan.style == ReplyStyle::SHORT ? settings.short_reply_chars :
                        plan.style == ReplyStyle::NORMAL                           ? settings.normal_reply_chars :
                                                                                     settings.long_reply_chars;
    const auto bubbles =
        split_bubbles(fit_length(clean_reply(final_text(*written), info, settings.aliases, settings.max_reply_bytes), budget));
    if (bubbles.empty()) {
        log::warn("writer returned an empty reply for {}", describe(chat));
        co_return;
    }
    if (observer) observer(chat, plan, bubbles);

    for (usize i = 0; i < bubbles.size(); ++i) {
        if (realtime) co_await lighter::sleep(typing_delay(bubbles[i], settings.typing_speed, rng));
        onebot::Message message;
        if (i == 0 && plan.quote && chat.type == "group") message.push_back(onebot::reply_segment(plan.target));
        message.push_back(onebot::text_segment(bubbles[i]));
        if (!co_await send_and_record(this, chat, std::move(message))) break;
    }
    note_spoke(state.flow, now());
}

} // namespace lilia::chat
