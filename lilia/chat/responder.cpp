#include "responder.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
#include <utility>
#include <variant>

#include <glaze/json.hpp>

#include <lighter/async/io/watcher.h>
#include <lighter/async/runtime/timeout.h>
#include <lighter/async/runtime/when.h>

#include <xmake/bin2obj/facts_md.hpp>
#include <xmake/bin2obj/self_facts_md.hpp>
#include <xmake/bin2obj/persona_md.hpp>
#include <xmake/bin2obj/planner_md.hpp>
#include <xmake/bin2obj/select_md.hpp>
#include <xmake/bin2obj/sticker_pick_md.hpp>
#include <xmake/bin2obj/writer_md.hpp>
#include <xmake/version/lilia.hpp>

#include <lilia/chat/delivery.h>
#include <lilia/chat/schedule.h>
#include <lilia/chat/trigger.h>
#include <lilia/chat/typos.h>
#include <lilia/log.h>
#include <lilia/onebot/message.h>
#include <lilia/net/codec.h>
#include <lilia/text.h>

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
/// Learned phrases shown to the writer as a sample of how the groups talk.
constexpr usize k_voice_samples = 8;
/// Recent targets whose reply she remembers.
constexpr usize k_replied_memory = 20;
/// Expression selection: too few candidates skip the call; the selector sees
/// this many recent lines.
constexpr usize k_min_candidates = 5;
constexpr usize k_selector_context = 10;
/// MaiBot's fallback sample (without embeddings) and its MMR weighting.
constexpr usize k_fallback_draw = 5;
constexpr usize k_min_repeated = 10;
constexpr double k_mmr_lambda = 0.85;
/// Facts: notes for this many people per turn; evidence is the person's last
/// few messages (at least this many characters) plus the last lines of chat.
constexpr usize k_people_notes = 3;
constexpr usize k_fact_own_messages = 3;
constexpr usize k_min_fact_evidence = 6;
constexpr usize k_fact_context = 8;
/// Her own facts: the closest this many to the last `k_self_context` lines
/// (plus the newest few).
constexpr usize k_self_facts = 6;
constexpr usize k_self_context = 4;
/// Poked, she pokes back this often, at most once per person in this many seconds.
constexpr double k_poke_back_chance = 0.7;
/// Messages looked at for a 复读 chain.
constexpr usize k_repeat_lookback = 8;
/// After she laughed (草, www), a leading laugh is dropped for this long.
constexpr i64 k_laugh_gap_seconds = 600;
constexpr i64 k_poke_back_seconds = 60;
/// MaiBot's response_splitter: more pieces than this go out as one message;
/// the rest are packed into at most this many messages.
constexpr usize k_max_pieces = 8;
constexpr usize k_max_messages = 3;
/// MaiBot's planner loop: rounds per turn and waits in a row (its limits);
/// replies per turn is lilia's own brake.
constexpr usize k_max_rounds = 8;
constexpr usize k_max_waits = 3;
constexpr usize k_max_replies = 3;
/// After a reconnect: this many of each group's newest messages are read, and
/// whoever called her this recently is still answered.
constexpr usize k_catch_up_messages = 100;
constexpr i64 k_late_answer_seconds = 1800;
/// At a trim of the log window, the planner's analyses of this many latest
/// turns stay; older turns keep only their tool calls and results.
constexpr usize k_kept_analyses = 6;
/// A failing planner model is skipped (fallbacks answer) for this long.
constexpr i64 k_planner_rest_seconds = 600;
/// Web searches the planner may run per turn.
constexpr usize k_max_web_searches = 2;

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

/// A string argument of a tool call; empty when missing.
std::string string_input(const glz::generic &input, std::string_view key) {
    if (!input.is_object()) return {};
    const auto &object = input.get_object();
    const auto it = object.find(key);
    return it != object.end() && it->second.is_string() ? it->second.get_string() : std::string();
}

/// "21:05" in the chat's local time.
std::string clock_of(i64 unix_seconds, i32 utc_offset_minutes) { return format_local_time(unix_seconds, utc_offset_minutes).substr(11); }

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

/// Sends one message and records it as lilia's own; its id, or nullopt when
/// it could not be sent.
Task<std::optional<i64>> send_and_record(Responder *r, const store::ChatKey &chat, onebot::Message message) {
    auto segments_json = glz::write_json(message).value_or("[]");
    auto text = onebot::render_plain(message);
    auto plain = onebot::text_only(message);
    const auto id = co_await send_message(r, chat, std::move(message));
    if (!id) co_return std::nullopt;
    auto stored = r->store.add_message({
        .chat = chat,
        .message_id = *id,
        .sender_id = r->self.user_id,
        .sender_name = r->self_name(chat),
        .time = r->now(),
        .segments_json = std::move(segments_json),
        .text = std::move(text),
        .from_self = true,
    });
    if (stored.has_error()) log::error("cannot record sent message: {}", stored.error().message());
    // Her own lines join the next learning batch as [SELF] context.
    if (r->learner && chat.type == "group") r->learner->observe_own(chat.id, std::move(plain));
    co_return id;
}

/// Nothing to add, as a task (the other side of a when_all).
Task<std::string> no_text() { co_return std::string(); }

/// with_timeout takes lighter's error type; this hands the provider's
/// outcome through as a plain value instead.
Task<Result<provider::ProviderCallCompletion>> settle(lighter::Task<provider::ProviderCallCompletion, Error> call) {
    auto outcome = co_await std::move(call);
    co_return outcome;
}

/// Per-purpose deadline for one model call (including the client's retries).
std::chrono::seconds call_deadline(const Responder *r, std::string_view purpose) {
    const auto &model = r->config.model;
    return std::chrono::seconds(purpose == "search" ? model.search_timeout_seconds :
                                purpose == "writer" ? model.writer_timeout_seconds :
                                                      model.timeout_seconds);
}

/// Runs one model call, records its usage, and returns the completed output
/// items (nullopt on failure, with the error in `failure` when given).
Task<std::optional<std::vector<provider::OutputItem>>>
call_model(Responder *r, provider::Provider &provider, std::string model, std::string_view purpose, const store::ChatKey &chat,
           const provider::History &history, const std::vector<provider::ToolDefinition> &tools, Error *failure = nullptr) {
    std::vector<provider::OutputItem> items;
    const provider::StreamCallbacks callbacks{
        .on_item_completed = [&items](const provider::OutputItem &item) { items.push_back(item); },
    };
    const i64 started = r->now();
    // A request that never finishes must not leave her silent: past the
    // deadline it fails like any other error (and the writer falls back).
    auto settled = co_await lighter::with_timeout(settle(provider->complete(history, tools, callbacks)), call_deadline(r, purpose));
    auto completion = settled.has_error() ?
                          Result<provider::ProviderCallCompletion>(lighter::outcome_error(Error::api(
                              "timeout", std::format("{} call exceeded {}s", purpose, call_deadline(r, purpose).count()), "", true))) :
                          *std::move(settled);

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
        log::info("{} call for {}: {} in / {} out / {} cache read / {} cache write tokens", purpose, describe(chat), call.input_tokens,
                  call.output_tokens, call.cache_read_tokens, call.cache_write_tokens);
    }
    if (auto recorded = r->store.add_llm_call(call); recorded.has_error()) {
        log::error("cannot record model call: {}", recorded.error().message());
    }
    if (completion.has_error()) {
        if (failure) *failure = completion.error();
        co_return std::nullopt;
    }
    co_return std::move(items);
}

/// A planner-model call, then each planner fallback in turn while they fail.
/// A planner that is rate limited (a used-up subscription window) or failed
/// twice in a row is skipped for a while, so every call does not first wait
/// out its deadline.
Task<std::optional<std::vector<provider::OutputItem>>> call_planner(Responder *r, std::string_view purpose, const store::ChatKey &chat,
                                                                    const provider::History &history,
                                                                    const std::vector<provider::ToolDefinition> &tools) {
    std::optional<std::vector<provider::OutputItem>> items;
    if (r->planner_fallbacks.empty() || r->now() >= r->planner_down_until) {
        Error failure;
        items = co_await call_model(r, r->planner, r->config.model.planner_model, purpose, chat, history, tools, &failure);
        if (items) {
            r->planner_failures = 0;
        } else if (!r->planner_fallbacks.empty() && (failure.status == 429 || ++r->planner_failures >= 2)) {
            r->planner_down_until = r->now() + k_planner_rest_seconds;
            r->planner_failures = 1; // after the rest, one more failure is enough
            log::warn("planner {} is resting for {} min; the fallbacks answer meanwhile", r->config.model.planner_model,
                      k_planner_rest_seconds / 60);
        }
    }
    for (auto &fallback : r->planner_fallbacks) {
        if (items) break;
        log::warn("planner fallback for {}: {}", describe(chat), fallback.model);
        items = co_await call_model(r, fallback.provider, fallback.model, purpose, chat, history, tools);
    }
    co_return items;
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
        const i64 quiet =
            flow.addressed || flow.named ? std::min<i64>(r->config.chat.debounce_seconds, 2) : r->config.chat.debounce_seconds;
        const i64 wait = flow.last_arrival + quiet - r->now();
        if (wait <= 0 || r->now() - started >= k_max_debounce_seconds) co_return;
        co_await lighter::sleep(std::chrono::seconds(wait));
    }
}

i64 random_seconds(Responder *r, u32 low, u32 high) {
    return std::uniform_int_distribution<i64>(std::min(low, high), std::max(low, high))(r->rng);
}

/// One OneBot action through the live session; false when it is gone or
/// the call failed.
Task<bool> call_action(Responder *r, std::string action, std::string params) {
    auto session = r->session;
    if (!session || !session->is_open()) co_return false;
    auto result = co_await session->call(action, std::move(params));
    if (result.has_error()) {
        log::warn("{} failed: {}", action, result.error().message());
        co_return false;
    }
    co_return true;
}

/// Joins a 复读 chain: the same text again after a moment.
bool muted(Responder *r, const store::ChatKey &chat) { return r->chats[chat].muted_until > r->now(); }

Task<> join_repeat(Responder *r, store::ChatKey chat, std::string text) {
    if (r->realtime) co_await lighter::sleep(std::chrono::seconds(random_seconds(r, 1, 4)));
    if (muted(r, chat) || !co_await send_and_record(r, chat, {onebot::text_segment(text)})) co_return;
    auto &state = r->chats[chat];
    note_spoke(state.flow);
    if (r->observer) r->observer(chat, {.reply = true, .intent = "复读"}, {text});
    log::info("{} joins the 复读: {}", describe(chat), text);
}

/// Poked in an allowed group: usually she pokes back after a moment, the
/// way people do; at most once a minute per person, never while asleep.
Task<> poke_back(Responder *r, i64 group, i64 user) {
    const store::ChatKey chat{.type = "group", .id = group};
    auto &state = r->chats[chat];
    const i64 now = r->now();
    if (now - state.poked_back[user] < k_poke_back_seconds || muted(r, chat)) co_return;
    if (presence_at(r->config.chat.schedule, now, r->config.chat.utc_offset_minutes).presence == Presence::ASLEEP) co_return;
    if (!std::bernoulli_distribution(k_poke_back_chance)(r->rng)) co_return;
    state.poked_back[user] = now;
    co_await lighter::sleep(std::chrono::seconds(random_seconds(r, 1, 4)));
    if (co_await call_action(r, "send_poke", std::format(R"({{"group_id":{},"user_id":{}}})", group, user)))
        log::info("{} pokes {} back", describe(chat), user);
}

/// Waits until `until` (unix seconds), looking every half minute; returns
/// early once someone calls her, if `called_ends_it`.
Task<> wait_until(Responder *r, const store::ChatKey &chat, i64 until, bool called_ends_it) {
    for (;;) {
        const i64 left = until - r->now();
        if (left <= 0 || (called_ends_it && r->chats[chat].flow.addressed)) co_return;
        co_await lighter::sleep(std::chrono::seconds(std::min<i64>(left, 30)));
    }
}

/// Her day, before a turn: asleep she waits until morning (whoever called
/// her is answered then); busy she waits until she is back, or answers a
/// call a few minutes late; awake she takes a moment to read. Sets the note
/// the next turn's planner and writer see.
Task<> live_presence(Responder *r, const store::ChatKey &chat) {
    const auto &schedule = r->config.chat.schedule;
    const auto offset = r->config.chat.utc_offset_minutes;
    for (;;) {
        auto &state = r->chats[chat];
        const auto presence = presence_at(schedule, r->now(), offset);
        if (presence.presence == Presence::ASLEEP) {
            log::info("{} is asleep until {}", describe(chat), clock_of(presence.until, offset));
            co_await wait_until(r, chat, presence.until, false);
            state.presence_note = std::format(
                "她刚睡醒：下面的新消息大多是她睡着时（{}–{}）发的。有人叫她的话现在回，可以说刚醒、刚看到；已经没意义的就不用回。",
                schedule.sleep_start, schedule.sleep_end);
            continue;
        }
        if (presence.presence == Presence::BUSY) {
            if (!state.flow.addressed) {
                log::info("{}: busy until {}", describe(chat), clock_of(presence.until, offset));
                co_await wait_until(r, chat, presence.until, true);
                if (!state.flow.addressed) {
                    state.presence_note = "她刚才在忙，现在才回来看群。";
                    continue;
                }
            }
            // Called while busy: she sees it a few minutes later.
            const i64 late =
                std::min(presence.until - r->now(), random_seconds(r, schedule.busy_reply_min_seconds, schedule.busy_reply_max_seconds));
            log::info("{}: busy, answers in {}s", describe(chat), late);
            co_await lighter::sleep(std::chrono::seconds(std::max<i64>(late, 0)));
            state.presence_note = "她刚才在忙，现在才看到有人叫她。";
        }
        break;
    }
    if (!schedule.enabled) co_return;
    // A moment to read before reacting; quicker when someone is talking to her.
    const auto &flow = r->chats[chat].flow;
    const bool called = flow.addressed || flow.named;
    co_await lighter::sleep(std::chrono::seconds(called ?
                                                     random_seconds(r, schedule.read_called_min_seconds, schedule.read_called_max_seconds) :
                                                     random_seconds(r, schedule.read_min_seconds, schedule.read_max_seconds)));
}

/// What get_group_member_info says about her in one group.
struct MemberInfo {
    std::string card;
    i64 shut_up_timestamp = 0; ///< muted until then (unix seconds); 0: not muted
};

/// Her group card and whether she is muted in an allowed group, asked once
/// per connection.
Task<> learn_self(Responder *r, i64 group) {
    auto session = r->session;
    if (!session || !session->is_open()) co_return;
    auto data = co_await session->call("get_group_member_info",
                                       std::format(R"({{"group_id":{},"user_id":{},"no_cache":true}})", group, r->self.user_id));
    if (data.has_error()) {
        log::warn("get_group_member_info for group {} failed: {}", group, data.error().message());
        co_return;
    }
    MemberInfo info;
    if (glz::read<onebot::k_read_opts>(info, *data)) co_return;
    auto &state = r->chats[{.type = "group", .id = group}];
    state.self_card = info.card;
    state.muted_until = info.shut_up_timestamp > r->now() ? info.shut_up_timestamp : 0;
    log::info("group {}: her card is \"{}\"{}", group, info.card, state.muted_until ? ", muted" : "");
}

Task<> turn_worker(Responder *r, store::ChatKey chat) {
    // Leave the reader loop's call stack before doing any work.
    co_await lighter::yield();
    for (;;) {
        auto &state = r->chats[chat];
        if (muted(r, chat)) {
            // What came in while she could not speak is stale once she can.
            state.flow.unseen.clear();
            state.flow.addressed.reset();
            state.flow.named.reset();
        }
        if (!should_plan(state.flow, r->flow_settings(), r->now())) {
            // Held back by the idle backoff: look again when it ends, with
            // whatever came in meanwhile, instead of leaving them unseen.
            if (const auto due = next_check(state.flow, r->flow_settings(), r->now()); due && r->realtime) {
                co_await lighter::sleep(std::chrono::seconds(std::max<i64>(*due - r->now(), 1)));
                continue;
            }
            state.busy = false;
            co_return;
        }
        if (r->realtime) {
            co_await live_presence(r, chat);
            co_await debounce(r, chat);
        }
        co_await r->run_turn(chat);
    }
}

/// Stores a message of an allowed chat with its quote, indexes it and starts
/// its enrichment; `replies_to_self` tells whether it quotes her.
store::StoredMessage record_event(Responder *r, const onebot::MessageEvent &message, bool &replies_to_self) {
    auto stored = to_stored(message);
    // A quote is part of what was said: "[回复 小明: …] 内容".
    std::string quote;
    replies_to_self = false;
    if (const auto target = onebot::reply_target(message.message)) {
        auto quoted = r->store.message(stored.chat, *target);
        const auto *found = quoted.has_value() && *quoted ? &**quoted : nullptr;
        replies_to_self = found && found->from_self;
        quote = quote_prefix(found, r->self_name(stored.chat));
        stored.text = quote + stored.text;
    }
    if (auto added = r->store.add_message(stored); added.has_error()) {
        log::error("cannot store message: {}", added.error().message());
    } else if (r->memory) {
        r->spawn(r->memory->index(stored.chat));
    }
    if (r->archiver && r->session && archive::needs_enrichment(message.message, r->archiver->config.enabled)) {
        r->archiver->track(stored.chat, stored.message_id);
        r->spawn(r->archiver->archive(r->session, stored, message.message, quote));
    }
    return stored;
}

/// What came in while she was away (a restart, a lost connection) since
/// `since` is stored as history; whoever called her in the last half hour is
/// answered late.
Task<> catch_up(Responder *r, i64 group, i64 since) {
    auto session = r->session;
    const store::ChatKey chat{.type = "group", .id = group};
    if (!session || !session->is_open()) co_return;
    const auto history = co_await onebot::group_history(session, group, k_catch_up_messages);
    usize added = 0;
    i64 first = 0;
    for (const auto &message : history) {
        if (message.time < since || message.user_id == r->self.user_id || !chat_allowed(message, r->config.chat)) continue;
        if (auto known = r->store.message(chat, message.message_id); !known.has_value() || *known) continue;
        bool replies_to_self = false;
        auto stored = record_event(r, message, replies_to_self);
        if (first == 0) first = message.time;
        ++added;
        const auto reason = classify(message, r->self.user_id, replies_to_self, r->config.chat);
        auto &state = r->chats[chat];
        if ((reason == TriggerReason::MENTION || reason == TriggerReason::REPLY) && r->now() - message.time <= k_late_answer_seconds &&
            !muted(r, chat)) {
            state.flow.addressed = message.message_id;
            state.flow.unseen.push_back(std::move(stored));
        }
    }
    if (added == 0) co_return;
    log::info("{}: caught up on {} message(s) from while she was away", describe(chat), added);
    auto &state = r->chats[chat];
    state.presence_note = std::format("她刚上线：{} 以后有些消息是她不在线时发的，已经过去的事不用回；有人叫过她的话现在回，可以说刚看到。",
                                      clock_of(first, r->config.chat.utc_offset_minutes));
    if (!state.flow.unseen.empty() && !state.busy) {
        state.busy = true;
        r->spawn(turn_worker(r, chat));
    }
}

/// On every connection: her card and mute state in the group, then what she missed.
Task<> rejoin(Responder *r, i64 group) {
    // Read before anything new is stored. A chat she has never seen starts
    // from now, not from its history.
    auto newest = r->store.recent_messages({.type = "group", .id = group}, 1);
    co_await learn_self(r, group);
    if (newest.has_value() && !newest->empty()) co_await catch_up(r, group, newest->back().time);
}

Task<> handle_admin(Responder *r, store::StoredMessage command, std::string arguments) {
    std::string reply;
    if (arguments == "status") {
        const i64 day = local_day_start(r->now(), r->config.chat.utc_offset_minutes);
        auto used = r->store.tokens_since(day, store::TokenScope::CHATS);
        auto background = r->store.tokens_since(r->now() - 86'400, store::TokenScope::BACKGROUND);
        reply = std::format("lilia {} ({}{})\n模型：{} / 规划 {}\n今日聊天 token：{} / {}\n24 小时学习 token：{} / {}\n已允许群：{} 个",
                            xmake::version::version, xmake::version::commit.substr(0, 8), xmake::version::dirty ? "-dirty" : "",
                            r->config.model.model, r->config.model.planner_model, used.has_value() ? *used : 0,
                            r->config.chat.daily_token_budget, background.has_value() ? *background : 0,
                            r->config.learning.daily_token_budget, r->config.chat.groups.size());
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

std::vector<store::Expression> sample_fallback(std::vector<store::Expression> pool, std::mt19937 &rng) {
    if (pool.empty()) return {};
    // MaiBot's fallback without embeddings: weights mapped from counts onto
    // 1..5; five from everything, five from patterns learned more than once.
    const auto [low, high] = std::ranges::minmax(pool | std::views::transform(&store::Expression::count));
    const auto weight = [&](const store::Expression &row) {
        return high == low ? 1.0 : 1.0 + 4.0 * static_cast<double>(row.count - low) / static_cast<double>(high - low);
    };
    std::vector<store::Expression> picked;
    const auto draw = [&](const std::vector<const store::Expression *> &from, usize count) {
        std::vector<double> weights;
        for (const auto *row : from) weights.push_back(std::ranges::contains(picked, row->id, &store::Expression::id) ? 0.0 : weight(*row));
        for (usize n = 0; n < count; ++n) {
            if (std::ranges::all_of(weights, [](double w) { return w == 0; })) break;
            std::discrete_distribution<usize> pick(weights.begin(), weights.end());
            const auto index = pick(rng);
            picked.push_back(*from[index]);
            weights[index] = 0;
        }
    };
    std::vector<const store::Expression *> everything;
    std::vector<const store::Expression *> repeated;
    for (const auto &row : pool) {
        everything.push_back(&row);
        if (row.count > 1) repeated.push_back(&row);
    }
    draw(everything, k_fallback_draw);
    if (repeated.size() >= k_min_repeated) draw(repeated, k_fallback_draw);
    return picked;
}

std::vector<std::string> sample_voice(store::Store &store, usize count) {
    // Over-fetch: some styles ("句尾加…") have no quoted phrase.
    auto rows = store.random_expressions(0, count * 2);
    if (rows.has_error()) return {};
    std::vector<std::string> phrases;
    for (const auto &row : *rows) {
        if (auto phrase = quoted_phrase(row.style); !phrase.empty() && !std::ranges::contains(phrases, phrase))
            phrases.push_back(std::move(phrase));
        if (phrases.size() >= count) break;
    }
    return phrases;
}

namespace {

std::string speaker(const Responder *r, const store::StoredMessage &message) {
    if (message.from_self) return r->config.chat.name + "（她自己）";
    return message.sender_name.empty() ? std::to_string(message.sender_id) : message.sender_name;
}

/// MaiBot's expression selection, after the planner: candidates retrieved by
/// embedding against the planner's intent and the chat (MMR for variety), or a
/// weighted sample without embeddings; then a selector call with the recent
/// chat picks 0 to `max_selected`. Picks are marked active.
Task<std::vector<std::string>> select_habits(Responder *r, store::ChatKey chat, std::vector<store::StoredMessage> log,
                                             store::StoredMessage target, std::string intent) {
    const auto &learning = r->config.learning;
    if (!learning.enabled) co_return std::vector<std::string>{};
    auto rows = r->store.expressions(0);
    if (rows.has_error()) co_return std::vector<std::string>{};
    std::erase_if(*rows, [&](const store::Expression &row) {
        return (learning.checked_only && !row.checked) || !learn::shares(learning.expression_groups, chat.id, row.chat.id);
    });
    if (rows->size() < k_min_candidates) co_return std::vector<std::string>{};

    std::vector<std::string> recent;
    for (usize i = log.size() > k_selector_context ? log.size() - k_selector_context : 0; i < log.size(); ++i) {
        recent.push_back(speaker(r, log[i]) + ": " + log[i].text);
    }

    std::vector<store::Expression> candidates;
    if (r->memory) {
        std::vector<const store::Expression *> embedded;
        std::vector<std::vector<float>> vectors;
        for (const auto &row : *rows) {
            if (row.vector.empty() || row.model != r->memory->config.embedding_model) continue;
            embedded.push_back(&row);
            vectors.push_back(memory::decode_vector(row.vector));
        }
        if (embedded.size() >= k_min_candidates) {
            std::string query = std::format("表达选择意图：{}\n要回应的消息：{}\n最近的聊天：\n", intent, target.text);
            for (usize i = recent.size() > 3 ? recent.size() - 3 : 0; i < recent.size(); ++i) query += recent[i] + "\n";
            auto embedded_query = co_await r->memory->embed({query});
            if (embedded_query.has_value() && !embedded_query->empty()) {
                for (const auto index : learn::mmr(vectors, embedded_query->front(), learning.candidates, k_mmr_lambda)) {
                    candidates.push_back(*embedded[index]);
                }
            }
        }
    }
    if (candidates.empty()) candidates = sample_fallback(*std::move(rows), r->rng);
    if (candidates.empty()) co_return std::vector<std::string>{};

    learn::SelectionInput input{
        .recent = std::move(recent),
        .target = speaker(r, target) + ": " + target.text,
        .intent = intent,
        .max_picks = learning.max_selected,
    };
    for (const auto &row : candidates) input.candidates.push_back(std::format("当{}时，可以{}", row.situation, row.style));
    const auto prompt = fill_template(embedded(xmake::select_md), {{"name", r->config.chat.name}});
    const auto request = learn::build_selection_request(prompt, input);
    const auto items = co_await call_planner(r, "select", chat, request.history, request.tools);
    if (!items) co_return std::vector<std::string>{};
    std::vector<std::string> picked;
    std::vector<i64> ids;
    for (const auto index : learn::parse_picks(*items, "select_expressions", "ids", candidates.size(), learning.max_selected)) {
        picked.push_back(input.candidates[index]);
        ids.push_back(candidates[index].id);
    }
    std::ignore = r->store.touch_expressions(ids, r->now());
    log::info("{} picks {} of {} learned expression(s)", describe(chat), picked.size(), candidates.size());
    co_return picked;
}

/// Slang meanings for the terms others use in `log` (MaiBot's jargon
/// reference); empty when slang learning is off or nothing matches.
std::string jargon_notes(Responder *r, const store::ChatKey &chat, const std::vector<store::StoredMessage> &log) {
    if (!r->learner || !r->config.learning.jargon) return {};
    auto terms = r->store.jargon_terms();
    if (terms.has_error()) return {};
    if (!r->config.learning.jargon_groups.empty()) {
        std::erase_if(*terms, [&](const store::Jargon &term) {
            auto chats = r->store.jargon_chats(term.id);
            return term.created_by != "MANUAL" && chats.has_value() &&
                   std::ranges::none_of(*chats, [&](i64 id) { return learn::shares(r->config.learning.jargon_groups, chat.id, id); });
        });
    }
    auto frequent = r->store.frequent_terms(chat);
    return learn::jargon_reference(*terms, log, frequent.has_value() ? *frequent : std::vector<store::FrequentTerm>{},
                                   r->config.learning.jargon_matches);
}

} // namespace

namespace {

/// She has stickers she may send.
bool stickers_ready(Responder *r) {
    if (!r->stickers) return false;
    auto pool = r->store.stickers("registered");
    return pool.has_value() && !pool->empty();
}

/// Picks a registered sticker for `feeling` (a pick call over a weighted
/// sample) and sends it as a QQ sticker; false when none fits, she sent one
/// too recently, or sending failed.
Task<bool> send_sticker(Responder *r, store::ChatKey chat, std::string feeling, std::vector<store::StoredMessage> log) {
    auto &state = r->chats[chat];
    if (!r->stickers || r->now() - state.last_sticker < r->config.stickers.cooldown_seconds) co_return false;
    auto pool = r->store.stickers("registered");
    if (pool.has_error() || pool->empty()) co_return false;
    const auto candidates = sticker::sample(*std::move(pool), r->config.stickers.candidates, r->rng);
    std::vector<std::string> recent;
    for (usize i = log.size() > k_selector_context ? log.size() - k_selector_context : 0; i < log.size(); ++i) {
        recent.push_back(speaker(r, log[i]) + ": " + log[i].text);
    }
    const auto prompt = fill_template(embedded(xmake::sticker_pick_md), {{"name", r->config.chat.name}});
    const auto request = sticker::build_pick_request(prompt, recent, feeling, candidates);
    const auto items = co_await call_planner(r, "sticker", chat, request.history, request.tools);
    if (!items) co_return false;
    const auto picks = learn::parse_picks(*items, "pick_sticker", "ids", candidates.size(), 1);
    if (picks.empty()) {
        log::info("{} found no sticker for \"{}\"", describe(chat), feeling);
        co_return false;
    }
    const auto &chosen = candidates[picks.front()];
    const auto bytes = r->stickers->read(chosen);
    if (!bytes) co_return false;

    onebot::Segment image{.type = "image"};
    image.data["file"] = "base64://" + net::base64_encode(*bytes);
    image.data["sub_type"] = i64{1};
    image.data["summary"] = "[动画表情]";
    const auto id = co_await send_message(r, chat, {std::move(image)});
    if (!id) co_return false;
    // The stored copy names the sticker instead of carrying its bytes.
    auto stored = r->store.add_message({
        .chat = chat,
        .message_id = *id,
        .sender_id = r->self.user_id,
        .sender_name = r->self_name(chat),
        .time = r->now(),
        .segments_json = std::format(R"([{{"type":"image","data":{{"file":"sticker:{}","sub_type":1}}}}])", chosen.sha1),
        .text = std::format("[表情: {}]", chosen.description),
        .from_self = true,
    });
    if (stored.has_error()) log::error("cannot record sent sticker: {}", stored.error().message());
    std::ignore = r->store.sticker_used(chosen.id, r->now());
    state.last_sticker = r->now();
    log::info("{} sent a sticker for \"{}\": [{}] {}", describe(chat), feeling, chosen.tags, chosen.description);
    co_return true;
}

/// What she knows about the person she answers, for the writer.
std::string known_about(Responder *r, const store::StoredMessage &target) {
    if (!r->memory || !r->config.memory.facts || target.from_self) return {};
    auto facts = r->store.facts(target.sender_id, r->config.memory.facts_per_person);
    return facts.has_value() ? memory::render_facts(*facts) : std::string();
}

/// MaiBot's person-fact writeback: after she answers someone, learn stable
/// facts from that person's own recent messages (checked against quotes).
Task<> learn_facts(Responder *r, store::ChatKey chat, store::StoredMessage target, std::vector<store::StoredMessage> log) {
    std::vector<store::StoredMessage> own;
    for (auto it = log.rbegin(); it != log.rend() && own.size() < k_fact_own_messages; ++it) {
        if (it->from_self || it->sender_id != target.sender_id) continue;
        // Facts quote the person, never the line they quoted.
        own.push_back(*it);
        own.back().text = std::string(without_quote(it->text));
    }
    std::ranges::reverse(own);
    usize evidence = 0;
    for (const auto &message : own) evidence += count_code_points(message.text);
    if (evidence < k_min_fact_evidence) co_return;
    const std::vector<store::StoredMessage> context(log.end() - static_cast<std::ptrdiff_t>(std::min(log.size(), k_fact_context)),
                                                    log.end());
    const auto prompt = fill_template(embedded(xmake::facts_md), {{"name", r->config.chat.name}});
    const auto request = memory::build_fact_request(prompt, speaker(r, target), own, context, r->config.chat.name);
    const auto items = co_await call_planner(r, "facts", chat, request.history, request.tools);
    if (!items) co_return;
    const auto facts = memory::parse_facts(*items, own);
    for (const auto &fact : facts) {
        const auto &source = own[fact.source];
        std::ignore = co_await r->memory->remember({
            .person_id = target.sender_id,
            .fact = fact.fact,
            .quote = fact.quote,
            .chat = chat,
            .message_id = source.message_id,
            .time = source.time,
        });
    }
}

/// The same writeback for herself: what she just said about herself (quoted,
/// newest wins on the same matter), so she stays consistent later.
Task<> learn_self_facts(Responder *r, store::ChatKey chat, std::vector<store::StoredMessage> own, std::vector<store::StoredMessage> log) {
    usize evidence = 0;
    for (const auto &message : own) evidence += count_code_points(message.text);
    if (evidence < k_min_fact_evidence) co_return;
    const std::vector<store::StoredMessage> context(log.end() - static_cast<std::ptrdiff_t>(std::min(log.size(), k_fact_context)),
                                                    log.end());
    const auto prompt = fill_template(embedded(xmake::self_facts_md), {{"name", r->config.chat.name}});
    const auto request = memory::build_fact_request(prompt, r->config.chat.name, own, context, r->config.chat.name);
    const auto items = co_await call_planner(r, "self facts", chat, request.history, request.tools);
    if (!items) co_return;
    for (const auto &fact : memory::parse_facts(*items, own)) {
        const auto &source = own[fact.source];
        std::ignore = co_await r->memory->remember(
            {
                .person_id = r->self.user_id,
                .fact = fact.fact,
                .quote = fact.quote,
                .chat = chat,
                .message_id = source.message_id,
                .time = source.time,
            },
            true);
    }
}

} // namespace

Responder::Responder(const Config &config, store::Store &store, provider::Provider planner, provider::Provider searcher,
                     provider::Provider writer, std::filesystem::path persona_path, Spawn spawn)
    : now([] { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }),
      config(config), store(store), planner(std::move(planner)), searcher(std::move(searcher)), writer(std::move(writer)),
      persona_path(std::move(persona_path)), persona(default_persona()), writer_rules(embedded(xmake::writer_md)), spawn(std::move(spawn)) {
}

void Responder::attach(std::shared_ptr<onebot::Session> live, onebot::LoginInfo login) {
    session = std::move(live);
    self = std::move(login);
    if (next_notice_id == 0) next_notice_id = -now() * 1000;
    for (const auto group : config.chat.groups) spawn(rejoin(this, group));
}

std::vector<store::StoredMessage> Responder::log_window(const store::ChatKey &chat) {
    auto &state = chats[chat];
    const auto &settings = config.chat;
    const auto size = [](const store::StoredMessage &message) { return message.text.size() + message.sender_name.size() + 24; };
    if (state.log_anchor) {
        if (auto window = store.messages_after(chat, *state.log_anchor, 2 * settings.context_messages + 1); window.has_value()) {
            usize bytes = 0;
            for (const auto &message : *window) bytes += size(message);
            if (window->size() <= 2 * settings.context_messages && bytes <= 2 * settings.context_bytes) return *std::move(window);
        }
    }
    auto recent = store.recent_messages(chat, settings.context_messages);
    if (!recent.has_value() || recent->empty()) {
        if (recent.has_error()) log::error("cannot read chat log: {}", recent.error().message());
        state.log_anchor.reset();
        return {};
    }
    usize bytes = 0;
    usize kept = 0;
    for (auto it = recent->rbegin(); it != recent->rend(); ++it, ++kept) {
        if (kept > 0 && bytes + size(*it) > settings.context_bytes) break;
        bytes += size(*it);
    }
    recent->erase(recent->begin(), recent->end() - static_cast<std::ptrdiff_t>(kept));
    const bool first = !state.log_anchor;
    state.log_anchor = recent->front().row_id - 1;
    // The planner's rounds about lines no longer shown go with them; older
    // turns keep their calls and results but lose their analyses. (The very
    // first anchor has nothing to trim.)
    if (!first) trim_records(state.records, *state.log_anchor, k_kept_analyses);
    return *std::move(recent);
}

std::string Responder::self_name(const store::ChatKey &chat) const {
    const auto it = chats.find(chat);
    return it != chats.end() && !it->second.self_card.empty() ? it->second.self_card : self.nickname;
}

void Responder::detach() noexcept { session.reset(); }

FlowSettings Responder::flow_settings() const { return {.talk_value = config.chat.talk_value}; }

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

    bool replies_to_self = false;
    auto stored = record_event(this, message, replies_to_self);
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

    if (state.muted_until > now()) {
        // Stored, but she cannot speak; what came in is stale once she can.
        state.flow.unseen.clear();
        state.flow.addressed.reset();
        state.flow.named.reset();
        return std::nullopt;
    }
    const auto reason = classify(message, self.user_id, replies_to_self, config.chat);
    if (reason != TriggerReason::NONE) {
        log::info("{} from {} in {}", to_string(reason), stored.sender_id, describe(stored.chat));
        // Her name alone may be talk about her; the planner decides.
        (reason == TriggerReason::ALIAS ? state.flow.named : state.flow.addressed) = stored.message_id;
    }
    const auto chat = stored.chat;
    note_arrival(state.flow, now());
    state.flow.unseen.push_back(std::move(stored));
    // QQ's 复读: people sending the same short line in a row; she may join.
    if (message.is_group()) {
        if (auto recent = store.recent_messages(chat, k_repeat_lookback); recent.has_value()) {
            if (const auto chain = repeat_chain(*recent); chain && chain->people >= 2 && chain->text != state.repeated) {
                const double chance = chain->people >= 4 ? 0.6 : chain->people == 3 ? 0.5 : 0.35;
                const auto presence = presence_at(config.chat.schedule, now(), config.chat.utc_offset_minutes).presence;
                if (presence == Presence::AWAKE && std::bernoulli_distribution(chance)(rng)) {
                    state.repeated = chain->text;
                    spawn(join_repeat(this, chat, chain->text));
                }
            }
        }
    }
    // Due now, or once the idle backoff ends (the worker waits for it).
    if (!should_plan(state.flow, flow_settings(), now()) && !next_check(state.flow, flow_settings(), now())) return std::nullopt;
    return chat;
}

namespace {

/// A person in a notice line: her name in the chat, the name on their newest
/// message, or their QQ number.
std::string person(Responder *r, const store::ChatKey &chat, i64 user) {
    if (user == r->self.user_id) return r->self_name(chat);
    auto name = r->store.sender_name(chat, user);
    return name.has_value() && !name->empty() ? *name : std::to_string(user);
}

/// "小明的「前十五个字…」", the message a notice is about; "一条消息" when
/// it is not stored.
std::string message_ref(Responder *r, const store::ChatKey &chat, i64 message_id) {
    auto found = r->store.message(chat, message_id);
    if (!found.has_value() || !*found) return "一条消息";
    const auto &message = **found;
    auto text = bounded_utf8(message.text, 45);
    if (text.size() < message.text.size()) text += "…";
    return std::format("{}的「{}」", message.from_self ? r->self_name(chat) : person(r, chat, message.sender_id), text);
}

/// Stores a notice line: context for her, never a turn of its own.
void record_notice(Responder *r, const store::ChatKey &chat, i64 time, std::string text) {
    auto added = r->store.add_message({
        .chat = chat,
        .message_id = r->next_notice_id--,
        .sender_id = 0,
        .sender_name = std::string(k_notice_sender),
        .time = time > 0 ? time : r->now(),
        .segments_json = "[]",
        .text = std::move(text),
    });
    if (added.has_error()) log::error("cannot store a notice: {}", added.error().message());
}

/// "15分钟" / "2小时" / "3天" for a mute duration in seconds.
std::string duration_text(i64 seconds) {
    if (seconds >= 86'400) return std::format("{}天", seconds / 86'400);
    if (seconds >= 3'600) return std::format("{}小时", seconds / 3'600);
    return std::format("{}分钟", std::max<i64>(seconds / 60, 1));
}

/// "[赞]" etc. for the faces of an emoji reaction; "表情" when none is known.
std::string reaction_faces(const std::optional<glz::generic_i64> &likes) {
    std::string out;
    if (likes && likes->is_array()) {
        for (const auto &like : likes->get_array()) {
            if (!like.is_object()) continue;
            const auto &object = like.get_object();
            const auto it = object.find("emoji_id");
            if (it == object.end()) continue;
            i64 id = -1;
            if (it->second.is_int64()) {
                id = it->second.get<i64>();
            } else if (it->second.is_string()) {
                const auto &text = it->second.get_string();
                std::from_chars(text.data(), text.data() + text.size(), id);
            }
            if (const auto name = onebot::face_name(id); !name.empty()) out += std::format("[{}]", name);
        }
    }
    return out.empty() ? std::string("表情") : out;
}

} // namespace

void Responder::recall(const store::ChatKey &chat, i64 message_id) {
    if (auto updated = store.update_message_text(chat, message_id, onebot::k_recalled_text); updated.has_error()) {
        log::error("cannot mark a recalled message: {}", updated.error().message());
    }
    auto &flow = chats[chat].flow;
    std::erase_if(flow.unseen, [&](const store::StoredMessage &message) { return message.message_id == message_id; });
    if (flow.addressed == message_id) flow.addressed.reset();
    if (flow.named == message_id) flow.named.reset();
    log::info("{}: message {} was recalled", describe(chat), message_id);
}

void Responder::notice(const onebot::NoticeEvent &notice) {
    if (!notice.group_id || !std::ranges::contains(config.chat.groups, *notice.group_id)) return;
    const store::ChatKey chat{.type = "group", .id = *notice.group_id};
    auto &state = chats[chat];
    const auto &type = notice.notice_type;
    const auto &sub = notice.sub_type;
    const auto operator_name = [&] { return person(this, chat, notice.operator_id.value_or(0)); };
    if (type == "notify" && sub == "poke" && notice.target_id) {
        record_notice(this, chat, notice.time,
                      std::format("{}戳了戳{}", person(this, chat, notice.user_id), person(this, chat, *notice.target_id)));
        if (notice.target_id == self.user_id && notice.user_id != self.user_id) spawn(poke_back(this, *notice.group_id, notice.user_id));
    } else if (type == "group_recall" && notice.message_id) {
        recall(chat, *notice.message_id);
    } else if (type == "group_ban") {
        const bool everyone = notice.user_id == 0;
        const bool lifted = sub == "lift_ban";
        if (everyone || notice.user_id == self.user_id) {
            // A whole-group mute has no duration; a day bounds a missed lift.
            const i64 seconds = everyone ? 86'400 : notice.duration.value_or(0);
            state.muted_until = lifted ? 0 : now() + seconds;
            log::info("{}: {}", describe(chat), lifted ? "she can speak again" : "she is muted");
        }
        record_notice(this, chat, notice.time,
                      everyone ? std::format("{}{}了全员禁言", operator_name(), lifted ? "解除" : "开启") :
                      lifted   ? std::format("{}解除了{}的禁言", operator_name(), person(this, chat, notice.user_id)) :
                                 std::format("{}把{}禁言了{}", operator_name(), person(this, chat, notice.user_id),
                                             duration_text(notice.duration.value_or(0))));
    } else if (type == "group_msg_emoji_like" && notice.message_id && notice.is_add.value_or(true)) {
        record_notice(this, chat, notice.time,
                      std::format("{}给{}贴了{}", person(this, chat, notice.user_id), message_ref(this, chat, *notice.message_id),
                                  reaction_faces(notice.likes)));
    } else if (type == "essence" && sub == "add" && notice.message_id) {
        record_notice(this, chat, notice.time,
                      std::format("{}把{}设为了精华消息", operator_name(), message_ref(this, chat, *notice.message_id)));
    } else if (type == "group_increase") {
        record_notice(this, chat, notice.time, std::format("{}加入了群聊", person(this, chat, notice.user_id)));
    } else if (type == "group_decrease" && sub != "kick_me") {
        record_notice(this, chat, notice.time,
                      std::format("{}{}了群聊", person(this, chat, notice.user_id), sub == "kick" ? "被移出" : "退出"));
    }
}

void Responder::on_event(onebot::Event event) {
    if (const auto *meta = std::get_if<onebot::MetaEvent>(&event)) {
        if (meta->meta_event_type == "lifecycle") log::info("onebot lifecycle: {}", meta->sub_type);
        return;
    }
    if (const auto *notice = std::get_if<onebot::NoticeEvent>(&event)) {
        if (session) this->notice(*notice);
        return;
    }
    // Until get_login_info answers, lilia does not know its own id.
    const auto *message = std::get_if<onebot::MessageEvent>(&event);
    if (!message || !session) return;
    // Every group may teach her how people talk; only allowed chats go further.
    if (learner) learner->observe(*message, self.user_id);
    if (stickers) stickers->observe(*message, self.user_id);
    const auto chat = ingest(*message);
    if (!chat) return;
    auto &state = chats[*chat];
    if (state.busy) return; // the running worker looks again after its turn
    state.busy = true;
    spawn(turn_worker(this, *chat));
}

namespace {

/// The planner's search_web tool: a call with only the provider-hosted
/// web_search tool, summarised in a few sentences; empty on failure.
Task<std::string> search_web(Responder *r, store::ChatKey chat, ChatInfo info, std::string query) {
    const auto search_history = build_search_history(query, r->now(), r->config.chat.utc_offset_minutes);
    const std::vector<provider::ToolDefinition> search_tools{{.kind = provider::ToolKind::WEB_SEARCH, .name = "web_search"}};
    std::string found;
    if (const auto items = co_await call_model(r, r->searcher, r->config.model.helper(), "search", chat, search_history, search_tools)) {
        found = clean_reply(final_text(*items), info, {}, 2'000);
    }
    log::info("{} searched the web for \"{}\": {} bytes", describe(chat), query, found.size());
    log::debug("search result: {}", found);
    co_return found;
}

/// What one reply came to.
struct Delivered {
    bool sent = false;
    std::string said; ///< her messages, joined
    /// Why nothing was attempted (the target was recalled), for the planner.
    std::string refused;
};

/// One reply as the planner asked for it: a sticker alone, or the writer's
/// text (with what the planner looked up this turn and the picked
/// expressions) sent the way MaiBot sends replies, plus a sticker when asked.
/// Records what she said.
Task<Delivered> deliver(Responder *r, store::ChatKey chat, Plan plan, std::vector<store::StoredMessage> log_lines,
                        store::StoredMessage target, ChatInfo info, i64 started, std::string remembered, std::string reference,
                        std::string self_notes, std::string presence) {
    auto &state = r->chats[chat];
    const auto &settings = r->config.chat;
    if (plan.sticker_only) {
        const auto feeling = plan.sticker.empty() ? plan.intent : plan.sticker + "：" + plan.intent;
        if (co_await send_sticker(r, chat, feeling, log_lines)) {
            if (r->observer) r->observer(chat, plan, {"[表情]"});
            note_spoke(state.flow);
            co_return Delivered{.sent = true, .said = "[表情包]"};
        }
        plan.style = ReplyStyle::SHORT; // nothing fitting: say it in words
    }

    std::string style_override;
    if (plan.style != ReplyStyle::LONG && std::bernoulli_distribution(std::clamp(settings.alt_style_probability, 0.0, 1.0))(r->rng)) {
        style_override = k_alt_styles[std::uniform_int_distribution<usize>(0, k_alt_styles.size() - 1)(r->rng)];
    }
    const auto habits = co_await select_habits(r, chat, log_lines, target, plan.intent.empty() ? plan.brief : plan.intent);
    const auto previous = std::ranges::find(state.replied, plan.target, &std::pair<i64, std::string>::first);
    const auto history = build_writer_history({
        .persona = r->persona,
        .rules = r->writer_rules,
        .info = info,
        .recent = log_lines,
        .target = target,
        .plan = plan,
        .style_override = style_override,
        .now = started,
        .utc_offset_minutes = settings.utc_offset_minutes,
        .log_bytes = std::numeric_limits<usize>::max(), // log_window() bounds it
        .reference = reference,
        .habits = habits,
        .remembered = remembered,
        .known = known_about(r, target),
        .self_notes = self_notes,
        .presence = presence,
        .previous_reply = previous == state.replied.end() ? std::string() : previous->second,
        .voice = r->config.learning.enabled ? sample_voice(r->store, k_voice_samples) : std::vector<std::string>{},
    });
    auto written = co_await call_model(r, r->writer, r->config.model.model, "writer", chat, history, {});
    for (auto &fallback : r->writer_fallbacks) {
        if (written && !final_text(*written).empty()) break;
        log::warn("writer fallback for {}: {}", describe(chat), fallback.model);
        written = co_await call_model(r, fallback.provider, fallback.model, "writer", chat, history, {});
    }
    if (!written) co_return Delivered{};
    log::debug("writer raw: {}", final_text(*written));
    const auto budget = !style_override.empty() || plan.style == ReplyStyle::SHORT ? settings.short_reply_chars :
                        plan.style == ReplyStyle::NORMAL                           ? settings.normal_reply_chars :
                                                                                     settings.long_reply_chars;
    // Emoji go unless people use them (they show up in learned expressions);
    // the feeling one stood for becomes a sticker, the way people answer.
    auto styles = r->store.expression_styles();
    const auto learned = styles.has_value() ? *std::move(styles) : std::vector<std::string>{};
    std::vector<char32_t> removed;
    const auto reply = strip_emoji(
        fit_length(clean_reply(strip_transcript(final_text(*written), log_lines), info, settings.aliases, settings.max_reply_bytes),
                   budget),
        emoji_in(learned), removed);
    if (!removed.empty() && plan.sticker.empty() && stickers_ready(r)) {
        for (const auto emoji : removed) {
            if (const auto feeling = emoji_feeling(emoji); !feeling.empty()) {
                plan.sticker = std::string(feeling);
                break;
            }
        }
    }

    // Casual replies go out the way MaiBot sends them: split at punctuation,
    // neighbours merged at random, a homophone typo now and then (sometimes
    // corrected right after), at most three messages. A long reply is on
    // purpose (an explanation, a copypasta): one message, untouched.
    const bool deliberate = plan.style == ReplyStyle::LONG && style_override.empty();
    std::vector<Segment> segments;
    if (deliberate) {
        for (auto &whole : whole_message(reply)) segments.push_back({.text = std::move(whole)});
    } else {
        // Every line, trimmed, blank ones dropped; no spaces around English words.
        auto casual = replace_haha(tighten_spacing(first_lines(reply, std::numeric_limits<usize>::max())), r->rng);
        // A laugh at the start of every message is a tic; one now and then is not.
        if (r->now() - state.last_laugh < k_laugh_gap_seconds) casual = drop_leading_laugh(casual);
        // Nothing but a laugh: people here send a sticker for that.
        if (only_laughs(casual) && stickers_ready(r) &&
            co_await send_sticker(r, chat, "笑死：" + (plan.intent.empty() ? std::string("觉得好笑") : plan.intent), log_lines)) {
            if (r->observer) r->observer(chat, plan, {"[表情]"});
            note_spoke(state.flow);
            co_return Delivered{.sent = true, .said = "[表情包]"};
        }
        segments = add_typos(split_segments(casual, r->rng), settings.typos, r->rng);
        if (segments.size() > k_max_pieces) {
            segments.clear();
            for (auto &whole : whole_message(casual)) segments.push_back({.text = std::move(whole)});
        }
        segments = pack_segments(std::move(segments), k_max_messages);
    }
    std::vector<std::string> bubbles;
    for (const auto &segment : segments) bubbles.push_back(segment.text);
    if (bubbles.empty()) {
        // Nothing but emoji: the sticker says it.
        if (!plan.sticker.empty() && co_await send_sticker(r, chat, plan.sticker + "：" + plan.intent, log_lines)) {
            if (r->observer) r->observer(chat, plan, {"[表情]"});
            note_spoke(state.flow);
            co_return Delivered{.sent = true, .said = "[表情包]"};
        }
        log::warn("writer returned an empty reply for {}", describe(chat));
        co_return Delivered{};
    }
    if (r->observer) r->observer(chat, plan, bubbles);

    std::optional<i64> sent_id;
    usize sent = 0;
    std::vector<store::StoredMessage> own; ///< what went out, for her own facts
    for (usize i = 0; i < segments.size(); ++i) {
        // Only later messages wait for "typing"; the first goes out at once.
        if (r->realtime && i > 0) co_await lighter::sleep(typing_delay(segments[i].text, settings.typing_speed));
        onebot::Message message;
        if (i == 0 && plan.quote && chat.type == "group" && plan.target > 0) {
            message.push_back(onebot::reply_segment(plan.target));
        } else if (segments[i].quote_previous && sent_id) {
            message.push_back(onebot::reply_segment(*sent_id));
        }
        message.push_back(onebot::text_segment(segments[i].text));
        sent_id = co_await send_and_record(r, chat, std::move(message));
        if (!sent_id) break;
        own.push_back({.chat = chat, .message_id = *sent_id, .time = r->now(), .text = segments[i].text, .from_self = true});
        ++sent;
    }
    if (sent == 0) co_return Delivered{};
    if (r->memory && r->config.memory.facts) r->spawn(learn_self_facts(r, chat, own, log_lines));
    if (!plan.sticker.empty()) {
        if (r->realtime) co_await lighter::sleep(std::chrono::milliseconds(800));
        std::ignore = co_await send_sticker(r, chat, plan.sticker + "：" + plan.intent, log_lines);
    }
    note_spoke(state.flow);
    if (r->memory && r->config.memory.facts && !target.from_self && !is_notice(target)) r->spawn(learn_facts(r, chat, target, log_lines));

    std::string said;
    for (usize i = 0; i < sent; ++i) said += (said.empty() ? "" : " ") + bubbles[i];
    if (laughs(said)) state.last_laugh = r->now();
    state.replied.emplace_back(plan.target, said);
    if (state.replied.size() > k_replied_memory) state.replied.pop_front();
    co_return Delivered{.sent = true, .said = std::move(said)};
}

/// The id of the first tool call in a planner response.
std::string first_call_id(const std::vector<provider::OutputItem> &items) {
    for (const auto &item : items) {
        if (const auto *call = std::get_if<provider::ToolCallItem>(&item)) return call->call.id;
    }
    return {};
}

/// The planner's analysis: the text it wrote before its tool call.
std::string analysis_of(const std::vector<provider::OutputItem> &items) {
    std::string text;
    for (const auto &item : items) {
        if (const auto *message = std::get_if<provider::AssistantMessageItem>(&item)) {
            for (const auto &part : message->parts) text += part.text;
        }
    }
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

/// The name of the first tool call in a planner response; empty when none.
std::string first_call_name(const std::vector<provider::OutputItem> &items) {
    for (const auto &item : items) {
        if (const auto *call = std::get_if<provider::ToolCallItem>(&item)) return call->call.name;
    }
    return {};
}

/// Results for every tool call of a planner response: `receipt` for the one
/// acted on, a note for any extra call (one tool per round).
std::vector<ToolOutcome> outcomes(const std::vector<provider::OutputItem> &items, std::string_view acted, std::string receipt) {
    std::vector<ToolOutcome> out;
    for (const auto &item : items) {
        const auto *call = std::get_if<provider::ToolCallItem>(&item);
        if (!call) continue;
        out.push_back(
            {.call_id = call->call.id, .receipt = call->call.id == acted ? receipt : std::string("没有执行：每次只能用一个工具。")});
    }
    return out;
}

} // namespace

Task<> Responder::run_turn(store::ChatKey chat) {
    auto &state = chats[chat];
    auto fresh = std::exchange(state.flow.unseen, {});
    auto addressed = std::exchange(state.flow.addressed, std::nullopt);
    auto named = std::exchange(state.flow.named, std::nullopt);
    if (fresh.empty() || state.muted_until > now()) co_return;

    const auto &settings = config.chat;
    const i64 started = now();
    const i64 day = local_day_start(started, settings.utc_offset_minutes);
    // Learning and sticker tagging have their own budget; they never silence her.
    auto used = store.tokens_since(day, store::TokenScope::CHATS);
    if (used.has_error()) log::error("cannot read token usage: {}", used.error().message());
    if (used.has_value() && *used >= settings.daily_token_budget) {
        log::warn("daily token budget exhausted ({} >= {})", *used, settings.daily_token_budget);
        if ((addressed || named) && state.budget_notice_day != day) {
            state.budget_notice_day = day;
            onebot::Message notice{onebot::text_segment("今天的额度用完了，明天再找我吧")};
            co_await send_and_record(this, chat, std::move(notice));
        }
        co_return;
    }

    // Let image descriptions of the new messages land before deciding, so she
    // reacts to what a picture shows rather than to "[图片]".
    const auto settle_images = [this, &chat, &settings](const std::vector<store::StoredMessage> &messages) -> Task<> {
        if (!archiver || !realtime) co_return;
        std::vector<i64> ids;
        for (const auto &message : messages) ids.push_back(message.message_id);
        co_await archiver->wait_for(chat, std::move(ids), std::chrono::seconds(settings.image_wait_seconds));
    };
    co_await settle_images(fresh);

    auto rows = log_window(chat);
    if (rows.empty()) co_return;
    const ChatInfo info{
        .chat = chat, .chat_name = state.name, .name = settings.name, .self_id = self.user_id, .self_name = self_name(chat)};
    const auto prompt = fill_template(embedded(xmake::planner_md), {{"name", settings.name},
                                                                    {"persona", persona},
                                                                    {"handoff", handoff_rule(settings.handoff == "reference")},
                                                                    {"memory_rule", memory_rule(memory != nullptr)},
                                                                    {"search_rule", search_rule(config.model.web_search)},
                                                                    {"sticker_rule", sticker_rule(stickers_ready(this))}});
    const auto tools = planner_tools(settings.handoff == "reference", memory != nullptr, config.model.web_search, stickers_ready(this));
    const u64 cycle = ++state.cycles;

    // Notes for this turn: in the tail of every round, never in the history.
    PlannerTail notes{.presence = std::exchange(state.presence_note, {}), .utc_offset_minutes = settings.utc_offset_minutes};
    if (memory && config.memory.facts) {
        std::vector<std::pair<i64, std::string>> people;
        for (auto it = fresh.rbegin(); it != fresh.rend() && people.size() < k_people_notes; ++it) {
            if (it->from_self || is_notice(*it) || std::ranges::contains(people, it->sender_id, &std::pair<i64, std::string>::first))
                continue;
            people.emplace_back(it->sender_id, speaker(this, *it));
        }
        notes.people = memory->people_notes(people);
    }
    notes.jargon = jargon_notes(this, chat, rows);
    if (memory) {
        // A summary the chat brings back, and what she said about herself that
        // bears on the last few lines: two lookups, side by side.
        std::string context;
        for (auto it = rows.end() - static_cast<std::ptrdiff_t>(std::min<usize>(rows.size(), k_self_context)); it != rows.end(); ++it)
            context += it->text + "\n";
        auto self_notes = config.memory.facts ? memory->self_notes(self.user_id, std::move(context), k_self_facts) : no_text();
        auto [recalled, own] = co_await lighter::when_all(memory->recall(chat, rows), std::move(self_notes));
        notes.recall = std::move(recalled);
        notes.self_notes = std::move(own);
    }

    const auto row_of = [&rows](i64 message_id) -> i64 {
        const auto it = std::ranges::find(rows, message_id, &store::StoredMessage::message_id);
        return it == rows.end() ? 0 : it->row_id;
    };
    // Facts about the lines new to this round, not instructions: the planner decides.
    const auto facts = [&](const std::optional<i64> &called, const std::optional<i64> &mentioned) {
        std::vector<std::string> out;
        if (chat.type != "group") return out;
        if (const i64 row = called ? row_of(*called) : 0) out.push_back(std::format("#{} 是叫她的（@ 了她或者回复了她的消息）。", row));
        if (const i64 row = mentioned ? row_of(*mentioned) : 0) out.push_back(std::format("#{} 提到了她的名字。", row));
        return out;
    };
    i64 first_new_row = 0;
    for (const auto &row : rows) {
        if (std::ranges::contains(fresh, row.message_id, &store::StoredMessage::message_id)) {
            first_new_row = row.row_id;
            break;
        }
    }
    i64 newest = fresh.back().message_id;

    // New lines during the turn simply join the history the next round shows.
    const auto refresh = [&]() -> Task<> {
        auto arrived = std::exchange(state.flow.unseen, {});
        addressed = std::exchange(state.flow.addressed, std::nullopt);
        named = std::exchange(state.flow.named, std::nullopt);
        co_await settle_images(arrived);
        if (auto latest = log_window(chat); !latest.empty()) rows = std::move(latest);
        if (!arrived.empty()) newest = arrived.back().message_id;
    };
    std::string remembered; // from search_history, for the writer
    std::string online;     // from search_web, for the writer
    usize web_searches = 0;
    const auto reply_to = [&](Plan plan) -> Task<Delivered> {
        auto target = std::ranges::find(rows, plan.target, &store::StoredMessage::message_id);
        if (target != rows.end() && target->from_self) {
            // She answers others, not herself; fall back to the newest message.
            plan.target = newest;
            target = std::ranges::find(rows, plan.target, &store::StoredMessage::message_id);
        }
        if (target == rows.end()) {
            log::warn("planner target {} is not in the chat log", plan.target);
            co_return Delivered{};
        }
        // Taken back, maybe after the log was read: not something to answer.
        if (auto current = store.message(chat, plan.target);
            target->text == onebot::k_recalled_text || (current.has_value() && *current && (*current)->text == onebot::k_recalled_text)) {
            log::info("{}: {} was recalled; not answering it", describe(chat), plan.target);
            co_return Delivered{.refused = "那条消息已经被撤回了，没有回。"};
        }
        log::info("{} replies to {} ({}, quote {}): {}", describe(chat), plan.target, style_name(plan.style), plan.quote,
                  plan.intent.empty() ? plan.brief : plan.intent);
        co_return co_await deliver(this, chat, plan, rows, *target, info, now(), remembered, online, notes.self_notes, notes.presence);
    };

    // MaiBot's planner loop: each round an analysis and one tool call; the
    // round, its result and the lines that came in meanwhile stay in the
    // chat's history for every later round and turn.
    usize searches = 0;
    usize waits = 0;
    usize replies = 0;
    i64 seen_row = 0;
    std::string last_analysis;
    std::string ended = "planner call failed";
    for (usize round = 0; round < k_max_rounds; ++round) {
        PlannerTail tail = notes;
        tail.first_new_row = round == 0 ? first_new_row : 0;
        if (round > 0) {
            for (const auto &row : rows) {
                if (row.row_id > seen_row && !row.from_self) {
                    tail.first_new_row = row.row_id;
                    break;
                }
            }
        }
        tail.facts = facts(addressed, named);
        tail.now = now();
        const i64 shown = rows.back().row_id;
        const auto planned = co_await call_planner(this, "planner", chat, render_session(prompt, info, rows, state.records, tail), tools);
        if (!planned) break;
        seen_row = shown;
        addressed.reset();
        named.reset();
        const auto analysis = analysis_of(*planned);
        last_analysis = analysis;
        if (!analysis.empty()) log::info("{} planner: {}", describe(chat), analysis);
        const auto call_id = first_call_id(*planned);
        const auto tool = first_call_name(*planned);
        // The round stays in the history with its result.
        const auto keep = [&](std::string receipt) {
            PlannerRecord record{.after_row = shown, .cycle = cycle, .output = *planned};
            if (!call_id.empty()) record.outcomes = outcomes(*planned, call_id, std::move(receipt));
            state.records.push_back(std::move(record));
        };

        // One she cannot use right now gets a receipt saying why (the tools
        // never change, which would void the cached prompt).
        std::string unavailable;
        if ((tool == "search_history" && searches >= config.memory.max_searches) ||
            (tool == "search_web" && web_searches >= k_max_web_searches)) {
            unavailable = "这一轮已经查过了，不能再查，根据现有的信息决定。";
        } else if ((tool == "wait" || tool == "stay_silent" || tool.empty()) && must_speak && replies == 0) {
            unavailable = "这次需要她说点什么，用 reply。";
        } else if (tool == "send_sticker" && now() - state.last_sticker < config.stickers.cooldown_seconds) {
            unavailable = "她刚发过表情包，隔一会儿再发；可以用文字回应，或者不说话。";
        }
        if (!unavailable.empty()) {
            keep(unavailable);
            co_await refresh();
            continue;
        }

        // A look into the history: its result goes back to the planner.
        if (memory && tool == "search_history") {
            const auto &call = std::get<provider::ToolCallItem>(*std::ranges::find_if(*planned, [](const provider::OutputItem &item) {
                                   return std::holds_alternative<provider::ToolCallItem>(item);
                               })).call;
            const auto hits = co_await memory->search(chat, string_input(call.input, "query"));
            const auto found = memory::render_hits(hits);
            if (!hits.empty()) remembered += found;
            keep(found);
            ++searches;
            continue;
        }
        // A look online: what it finds goes back to the planner, and to the
        // writer if she replies.
        if (config.model.web_search && tool == "search_web") {
            const auto &call = std::get<provider::ToolCallItem>(*std::ranges::find_if(*planned, [](const provider::OutputItem &item) {
                                   return std::holds_alternative<provider::ToolCallItem>(item);
                               })).call;
            const auto found = co_await search_web(this, chat, info, string_input(call.input, "query"));
            if (!found.empty()) online += (online.empty() ? "" : "\n") + found;
            keep(found.empty() ? std::string("没查到。") : "查到的：" + found);
            ++web_searches;
            continue;
        }

        auto plan = parse_plan(*planned, rows);
        plan.follow_up = replies > 0;
        plan.analysis = analysis;
        if (plan.repeat) {
            // 复读: the message again, as it is.
            const auto target = std::ranges::find(rows, plan.target, &store::StoredMessage::message_id);
            std::string receipt;
            if (target == rows.end() || target->from_self || is_notice(*target) || !repeatable(target->text)) {
                receipt = "这条不适合复读（只复读别人的短消息），没有发。";
            } else if (co_await send_and_record(this, chat, {onebot::text_segment(target->text)})) {
                ++replies;
                state.repeated = target->text;
                note_spoke(state.flow);
                if (observer) observer(chat, plan, {target->text});
                log::info("{} repeats {}: {}", describe(chat), plan.target, target->text);
                receipt = std::format("她复读了 #{}。", target->row_id);
            } else {
                receipt = "没发出去。";
            }
            keep(receipt);
            if (replies >= k_max_replies) {
                ended = "replied enough";
                break;
            }
            co_await refresh();
            continue;
        }
        if (plan.wait > 0) {
            // MaiBot allows three waits in a row, then the turn rests.
            if (++waits > k_max_waits) {
                keep("不能再等了，这一轮到此为止。");
                ended = "waited enough";
                break;
            }
            log::info("{} waits {}s: {}", describe(chat), plan.wait, plan.reason);
            if (observer) observer(chat, plan, {});
            const i64 before = now();
            if (pause) {
                co_await pause(chat, plan.wait);
            } else if (realtime) {
                co_await lighter::sleep(std::chrono::seconds(plan.wait));
            }
            keep(std::format("等了 {} 秒。", now() - before));
            co_await refresh();
            continue;
        }
        waits = 0;
        if (!plan.reply) {
            if (tool == "reply" || tool == "repeat" || tool == "send_sticker") {
                // A target that is not in the history: let it look again.
                keep("没找到这个编号的消息，没有发。");
                continue;
            }
            keep("好。");
            ended = tool.empty() ? (analysis.empty() ? std::string("no tool call") : analysis) : plan.reason;
            break;
        }

        const auto delivered = co_await reply_to(plan);
        if (!delivered.refused.empty()) {
            keep(delivered.refused);
            co_await refresh();
            continue;
        }
        if (!delivered.sent) {
            keep("没发出去。");
            ended = "the reply failed";
            break;
        }
        ++replies;
        std::string receipt = std::format("她回复了 #{}，说的话在上面的聊天记录里。", row_of(plan.target));
        keep(std::move(receipt));
        if (replies >= k_max_replies) {
            ended = "replied enough";
            break;
        }
        co_await refresh();
    }

    // Blind comparisons only: she must say something.
    if (must_speak && replies == 0 && ended != "the reply failed") {
        log::info("{} turn ended ({}) without a reply; replying anyway", describe(chat), ended);
        if ((co_await reply_to({.reply = true, .target = newest})).sent) ++replies;
    }
    if (replies == 0) {
        log::info("{} stays silent: {}", describe(chat), ended);
        note_silent(state.flow, now());
        if (observer) observer(chat, {.reason = ended, .analysis = last_analysis}, {});
    } else {
        log::info("{} ends the turn after {} repl{}: {}", describe(chat), replies, replies == 1 ? "y" : "ies", ended);
        if (observer) observer(chat, {.follow_up = true, .reason = ended, .analysis = last_analysis}, {});
    }
}

} // namespace lilia::chat
