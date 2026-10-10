#include "responder.h"

#include <algorithm>
#include <array>
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
constexpr usize k_decision_memory = 6;
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
constexpr i64 k_poke_back_seconds = 60;
/// She holds back once she wrote this many of the last `k_self_window` messages.
constexpr usize k_self_window = 10;
constexpr usize k_self_limit = 3;
/// MaiBot's response_splitter: more pieces than this go out as one message;
/// the rest are packed into at most this many messages.
constexpr usize k_max_pieces = 8;
constexpr usize k_max_messages = 3;
/// MaiBot's planner loop: rounds per turn and waits in a row (its limits);
/// replies per turn is lilia's own brake.
constexpr usize k_max_rounds = 8;
constexpr usize k_max_waits = 3;
constexpr usize k_max_replies = 3;

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

void remember_decision(Responder::ChatState &state, std::string decision) {
    state.decisions.push_back(std::move(decision));
    while (state.decisions.size() > k_decision_memory) state.decisions.pop_front();
}

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
        .sender_name = r->self.nickname,
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

/// Poked in an allowed group: usually she pokes back after a moment, the
/// way people do; at most once a minute per person, never while asleep.
Task<> poke_back(Responder *r, i64 group, i64 user) {
    const store::ChatKey chat{.type = "group", .id = group};
    auto &state = r->chats[chat];
    const i64 now = r->now();
    if (now - state.poked_back[user] < k_poke_back_seconds) co_return;
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
    // A moment to read before reacting.
    const bool called = r->chats[chat].flow.addressed.has_value();
    co_await lighter::sleep(std::chrono::seconds(called ?
                                                     random_seconds(r, schedule.read_called_min_seconds, schedule.read_called_max_seconds) :
                                                     random_seconds(r, schedule.read_min_seconds, schedule.read_max_seconds)));
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
        if (r->realtime) {
            co_await live_presence(r, chat);
            co_await debounce(r, chat);
        }
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
    const auto items = co_await call_model(r, r->planner, r->config.model.planner_model, "select", chat, request.history, request.tools);
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
    const auto items = co_await call_model(r, r->planner, r->config.model.planner_model, "sticker", chat, request.history, request.tools);
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
        .sender_name = r->self.nickname,
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
        if (!it->from_self && it->sender_id == target.sender_id) own.push_back(*it);
    }
    std::ranges::reverse(own);
    usize evidence = 0;
    for (const auto &message : own) evidence += count_code_points(message.text);
    if (evidence < k_min_fact_evidence) co_return;
    const std::vector<store::StoredMessage> context(log.end() - static_cast<std::ptrdiff_t>(std::min(log.size(), k_fact_context)),
                                                    log.end());
    const auto prompt = fill_template(embedded(xmake::facts_md), {{"name", r->config.chat.name}});
    const auto request = memory::build_fact_request(prompt, speaker(r, target), own, context, r->config.chat.name);
    const auto items = co_await call_model(r, r->planner, r->config.model.planner_model, "facts", chat, request.history, request.tools);
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
    const auto items =
        co_await call_model(r, r->planner, r->config.model.planner_model, "self facts", chat, request.history, request.tools);
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
    if (auto added = store.add_message(stored); added.has_error()) {
        log::error("cannot store message: {}", added.error().message());
    } else if (memory) {
        spawn(memory->index(stored.chat));
    }
    if (archiver && session && !archive::collect_media(message.message).empty()) {
        archiver->track(stored.chat, stored.message_id);
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
    if (const auto *notice = std::get_if<onebot::NoticeEvent>(&event)) {
        // Someone poked her in an allowed group.
        if (session && notice->notice_type == "notify" && notice->sub_type == "poke" && notice->group_id &&
            notice->target_id == self.user_id && notice->user_id != self.user_id &&
            std::ranges::contains(config.chat.groups, *notice->group_id))
            spawn(poke_back(this, *notice->group_id, notice->user_id));
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

/// What one reply came to.
struct Delivered {
    bool sent = false;
    std::string said; ///< her messages, joined
};

/// One reply as the planner asked for it: a sticker alone, or the writer's
/// text (after an optional web search and the expression selection) sent the
/// way MaiBot sends replies, plus a sticker when asked. Records what she said.
Task<Delivered> deliver(Responder *r, store::ChatKey chat, Plan plan, std::vector<store::StoredMessage> log_lines,
                        store::StoredMessage target, ChatInfo info, i64 started, std::string remembered, std::string self_notes,
                        std::string presence) {
    auto &state = r->chats[chat];
    const auto &settings = r->config.chat;
    if (plan.sticker_only) {
        const auto feeling = plan.sticker.empty() ? plan.intent : plan.sticker + "：" + plan.intent;
        if (co_await send_sticker(r, chat, feeling, log_lines)) {
            if (r->observer) r->observer(chat, plan, {"[表情]"});
            note_spoke(state.flow, r->now(), target.sender_id);
            co_return Delivered{.sent = true, .said = "[表情包]"};
        }
        plan.style = ReplyStyle::SHORT; // nothing fitting: say it in words
    }

    std::string style_override;
    if (plan.style != ReplyStyle::LONG && std::bernoulli_distribution(std::clamp(settings.alt_style_probability, 0.0, 1.0))(r->rng)) {
        style_override = k_alt_styles[std::uniform_int_distribution<usize>(0, k_alt_styles.size() - 1)(r->rng)];
    }
    std::string reference;
    if (plan.search && r->config.model.web_search) {
        // A dedicated call: writers asked to search "if needed" often don't.
        const auto search_history = build_search_history(info, target, plan, started, settings.utc_offset_minutes);
        const std::vector<provider::ToolDefinition> search_tools{{.kind = provider::ToolKind::WEB_SEARCH, .name = "web_search"}};
        if (const auto found =
                co_await call_model(r, r->searcher, r->config.model.planner_model, "search", chat, search_history, search_tools)) {
            reference = clean_reply(final_text(*found), info, {}, 2'000);
        }
        log::info("search for {} returned {} bytes", describe(chat), reference.size());
        log::debug("search reference: {}", reference);
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
        .log_bytes = settings.context_bytes,
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
    std::vector<std::string> learned;
    if (auto rows = r->store.expressions(0); rows.has_value()) {
        for (auto &row : *rows) learned.push_back(std::move(row.style));
    }
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
        const auto casual = replace_haha(tighten_spacing(first_lines(reply, std::numeric_limits<usize>::max())), r->rng);
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
            note_spoke(state.flow, r->now(), target.sender_id);
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
        if (i == 0 && plan.quote && chat.type == "group") {
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
    note_spoke(state.flow, r->now(), target.sender_id);
    if (r->memory && r->config.memory.facts && !target.from_self) r->spawn(learn_facts(r, chat, target, log_lines));

    std::string said;
    for (usize i = 0; i < sent; ++i) said += (said.empty() ? "" : " ") + bubbles[i];
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

    // Let image descriptions of the new messages land before deciding, so she
    // reacts to what a picture shows rather than to "[图片]".
    const auto settle_images = [this, &chat, &settings](const std::vector<store::StoredMessage> &messages) -> Task<> {
        if (!archiver || !realtime) co_return;
        std::vector<i64> ids;
        for (const auto &message : messages) ids.push_back(message.message_id);
        co_await archiver->wait_for(chat, std::move(ids), std::chrono::seconds(settings.image_wait_seconds));
    };
    co_await settle_images(fresh);

    auto recent = store.recent_messages(chat, settings.context_messages);
    if (recent.has_error()) log::error("cannot read chat log: {}", recent.error().message());
    auto log_lines = recent.has_value() ? *std::move(recent) : fresh;

    // MaiBot weighs "how much of the recent chat was me" heavily: once she has
    // said a lot lately, only being called (or answered right after her own
    // message) brings her back, and the planner is not even asked.
    if (!addressed) {
        const usize window = std::min(log_lines.size(), k_self_window);
        const auto recent_self =
            std::ranges::count_if(log_lines.end() - static_cast<std::ptrdiff_t>(window), log_lines.end(), &store::StoredMessage::from_self);
        // Whoever she is talking with (inside the conversation window) still
        // gets through; the brake is for chiming in on everyone else.
        const bool answering_her = state.flow.partner != 0 && started - state.flow.last_spoke <= settings.conversation_window_seconds &&
                                   std::ranges::contains(fresh, state.flow.partner, &store::StoredMessage::sender_id);
        if (static_cast<usize>(recent_self) >= k_self_limit && !answering_her) {
            log::info("{} holds back: she wrote {} of the last {} messages", describe(chat), recent_self, window);
            co_return;
        }
    }

    const ChatInfo info{.chat = chat, .chat_name = state.name, .name = settings.name, .self_id = self.user_id, .self_name = self.nickname};
    PlannerInput planner_input{
        .prompt = fill_template(embedded(xmake::planner_md), {{"name", settings.name},
                                                              {"persona", persona},
                                                              {"handoff", handoff_rule(settings.handoff == "reference")},
                                                              {"memory_rule", memory_rule(memory != nullptr)},
                                                              {"sticker_rule", sticker_rule(stickers_ready(this))}}),
        .info = info,
        .recent = log_lines,
        .addressed = addressed.has_value(),
        .now = started,
        .utc_offset_minutes = settings.utc_offset_minutes,
        .log_bytes = settings.context_bytes,
        .reference_handoff = settings.handoff == "reference",
        .decisions = {state.decisions.begin(), state.decisions.end()},
        .memory_search = memory != nullptr,
        .stickers = stickers_ready(this),
    };
    for (const auto &message : fresh) planner_input.fresh.push_back(message.message_id);
    if (memory && config.memory.facts) {
        std::vector<std::pair<i64, std::string>> people;
        for (auto it = fresh.rbegin(); it != fresh.rend() && people.size() < k_people_notes; ++it) {
            if (it->from_self || std::ranges::contains(people, it->sender_id, &std::pair<i64, std::string>::first)) continue;
            people.emplace_back(it->sender_id, speaker(this, *it));
        }
        planner_input.people = memory->people_notes(people);
        if (!planner_input.people.empty()) log::debug("{} notes on people:\n{}", describe(chat), planner_input.people);
    }
    if (memory) planner_input.recall = co_await memory->recall(chat, log_lines);
    planner_input.jargon = jargon_notes(this, chat, log_lines);
    planner_input.presence = std::exchange(state.presence_note, {});
    if (memory && config.memory.facts) {
        // What she said about herself that bears on the last few lines.
        std::string context;
        for (auto it = log_lines.end() - static_cast<std::ptrdiff_t>(std::min<usize>(log_lines.size(), k_self_context));
             it != log_lines.end(); ++it)
            context += it->text + "\n";
        planner_input.self_notes = co_await memory->self_notes(self.user_id, std::move(context), k_self_facts);
    }
    auto request = build_planner_request(planner_input);
    auto &history = request.history;
    auto &ids = request.ids;

    // MaiBot's planner loop: each round one tool call; after a reply or a
    // wait the planner sees what happened and the messages that came in.
    i64 newest = fresh.back().message_id;
    bool answered = false; // replied to whoever called her
    usize searches = 0;
    usize waits = 0;
    usize replies = 0;
    std::string remembered;
    // Messages that came in since the last look, numbered for the planner;
    // someone calling her meanwhile must be answered too.
    const auto arrivals = [&]() -> Task<std::string> {
        auto arrived = std::exchange(state.flow.unseen, {});
        if (auto called = std::exchange(state.flow.addressed, std::nullopt)) {
            addressed = called;
            answered = false;
        }
        if (arrived.empty()) co_return std::string();
        co_await settle_images(arrived);
        newest = arrived.back().message_id;
        if (auto latest = store.recent_messages(chat, settings.context_messages); latest.has_value()) log_lines = *std::move(latest);
        // Their descriptions may have landed since they were queued.
        for (auto &message : arrived) {
            if (const auto it = std::ranges::find(log_lines, message.message_id, &store::StoredMessage::message_id); it != log_lines.end())
                message = *it;
        }
        co_return number_messages(arrived, ids, settings.utc_offset_minutes);
    };
    const auto reply_to = [&](Plan plan) -> Task<Delivered> {
        auto target = std::ranges::find(log_lines, plan.target, &store::StoredMessage::message_id);
        if (target != log_lines.end() && target->from_self) {
            // She answers others, not herself; fall back to the newest message.
            plan.target = addressed.value_or(newest);
            target = std::ranges::find(log_lines, plan.target, &store::StoredMessage::message_id);
        }
        if (target == log_lines.end()) {
            log::warn("planner target {} is not in the chat log", plan.target);
            co_return Delivered{};
        }
        log::info("{} replies to {} ({}, quote {}, search {}): {}", describe(chat), plan.target, style_name(plan.style), plan.quote,
                  plan.search, plan.intent.empty() ? plan.brief : plan.intent);
        remember_decision(state, std::format("[{}] 回复了{}的「{}」：{}", clock_of(now(), settings.utc_offset_minutes),
                                             target->sender_name.empty() ? std::to_string(target->sender_id) : target->sender_name,
                                             bounded_utf8(target->text, 60), plan.intent.empty() ? plan.brief : plan.intent));
        auto delivered = co_await deliver(this, chat, plan, log_lines, *target, info, started, remembered, planner_input.self_notes,
                                          planner_input.presence);
        if (delivered.sent && addressed && plan.target == *addressed) answered = true;
        co_return delivered;
    };

    std::string ended = "planner call failed";
    for (usize round = 0; round < k_max_rounds; ++round) {
        auto offered = request.tools;
        if (!memory || searches >= config.memory.max_searches)
            std::erase_if(offered, [](const auto &tool) { return tool.name == "search_history"; });
        if (must_speak && replies == 0)
            std::erase_if(offered, [](const auto &tool) { return tool.name == "wait" || tool.name == "stay_silent"; });
        // Right after a sticker, no second one.
        if (now() - state.last_sticker < config.stickers.cooldown_seconds)
            std::erase_if(offered, [](const auto &tool) { return tool.name == "send_sticker"; });
        const auto planned = co_await call_model(this, planner, config.model.planner_model, "planner", chat, history, offered);
        if (!planned) break;
        for (const auto &item : *planned) provider::append_output_item(history, item);
        const auto call_id = first_call_id(*planned);

        // A look into the history: its result goes back to the planner.
        const auto search = std::ranges::find_if(*planned, [](const provider::OutputItem &item) {
            const auto *call = std::get_if<provider::ToolCallItem>(&item);
            return call && call->call.name == "search_history";
        });
        if (memory && search != planned->end()) {
            const auto &call = std::get<provider::ToolCallItem>(*search).call;
            const auto hits = co_await memory->search(chat, string_input(call.input, "query"));
            const auto found = memory::render_hits(hits);
            if (!hits.empty()) remembered += found;
            provider::append_tool_outcomes(history, outcomes(*planned, call.id, found));
            ++searches;
            continue;
        }

        auto plan = parse_plan(*planned, ids);
        plan.follow_up = replies > 0;
        if (plan.wait > 0) {
            // MaiBot allows three waits in a row, then the turn rests.
            if (++waits > k_max_waits) {
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
            const auto came = co_await arrivals();
            const auto waited = now() - before;
            provider::append_tool_outcomes(history, outcomes(*planned, call_id,
                                                             came.empty() ? std::format("等了 {} 秒，这期间没有新消息。", waited) :
                                                                            std::format("等了 {} 秒，这期间的新消息：\n{}", waited, came)));
            continue;
        }
        waits = 0;
        if (!plan.reply) {
            ended = plan.reason;
            break;
        }

        const auto delivered = co_await reply_to(plan);
        if (!delivered.sent) {
            ended = "the reply failed";
            break;
        }
        ++replies;
        if (replies >= k_max_replies) {
            ended = "replied enough";
            break;
        }
        const auto came = co_await arrivals();
        const auto index = static_cast<usize>(std::ranges::find(ids, plan.target) - ids.begin()) + 1;
        provider::append_tool_outcomes(history, outcomes(*planned, call_id,
                                                         std::format("{}回复了 #{}：「{}」\n{}", settings.name, index, delivered.said,
                                                                     came.empty() ? "这期间没有新消息。" : "这期间的新消息：\n" + came)));
    }

    // Ignoring someone who called her reads as broken, not as a choice.
    // (In blind comparisons she must say something; the planner can still
    // pick stay_silent, which it sees named in its prompt.)
    if (((addressed && !answered) || (must_speak && replies == 0)) && ended != "the reply failed") {
        const auto target = addressed && !answered ? *addressed : newest;
        log::info("{} turn ended ({}) without answering {}; replying anyway", describe(chat), ended, target);
        if ((co_await reply_to({.reply = true, .target = target, .follow_up = replies > 0})).sent) ++replies;
    }
    if (replies == 0) {
        log::info("{} stays silent: {}", describe(chat), ended);
        note_silent(state.flow, now());
        remember_decision(state, std::format("[{}] 没说话：{}", clock_of(started, settings.utc_offset_minutes), ended));
        if (observer) observer(chat, {.reason = ended}, {});
    } else {
        log::info("{} ends the turn after {} repl{}: {}", describe(chat), replies, replies == 1 ? "y" : "ies", ended);
        if (observer) observer(chat, {.follow_up = true, .reason = ended}, {});
    }
}

} // namespace lilia::chat
