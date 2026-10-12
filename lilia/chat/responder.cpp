#include "responder.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <utility>

#include <glaze/json.hpp>

#include <lighter/async/io/watcher.h>

#include <xmake/bin2obj/persona_md.hpp>
#include <xmake/bin2obj/writer_md.hpp>
#include <xmake/version/lilia.hpp>

#include <lilia/chat/detail/responder.h>
#include <lilia/chat/schedule.h>
#include <lilia/chat/trigger.h>
#include <lilia/log.h>

namespace lilia::chat {

using lighter::Task;
using namespace detail;

namespace {

constexpr i64 k_max_debounce_seconds = 15;

/// After a reconnect: this many of each group's newest messages are read, and
/// whoever called her this recently is still answered.
constexpr usize k_catch_up_messages = 100;
constexpr i64 k_late_answer_seconds = 1800;

/// At a trim of the log window, the planner's analyses of this many latest
/// turns stay; older turns keep only their tool calls and results.
constexpr usize k_kept_analyses = 6;

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

/// "21:05" in the chat's local time.
std::string clock_of(i64 unix_seconds, i32 utc_offset_minutes) { return format_local_time(unix_seconds, utc_offset_minutes).substr(11); }

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

bool muted(Responder *r, const store::ChatKey &chat) { return r->chats[chat].muted_until > r->now(); }

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
    // An @ without a name would read as a QQ number: hers shows as the
    // "@名字" the prompts describe, others' as the name stored for them.
    if (std::ranges::any_of(message.message, [](const onebot::Segment &segment) {
            return segment.type == "at" && onebot::data_string(segment, "name").value_or("").empty();
        })) {
        auto named = message;
        for (auto &segment : named.message) {
            const i64 user = segment.type == "at" ? onebot::data_integer(segment, "qq").value_or(0) : 0;
            if (user == 0 || !onebot::data_string(segment, "name").value_or("").empty()) continue;
            auto known = user == r->self.user_id ? Result<std::string>(r->self_name(stored.chat)) : r->store.sender_name(stored.chat, user);
            auto name = known.has_value() ? *std::move(known) : std::string();
            if (!name.empty()) segment.data["name"] = std::move(name);
        }
        stored = to_stored(named);
    }
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
    // What it no longer shows is summarised for it (after a restart too: the
    // lines before the first anchor may never have been).
    if (memory) spawn(memory->summarize_window(chat, *state.log_anchor, 2 * settings.context_messages));
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
    // Earlier windows may be due for judging now that the chat moved on.
    if (feedback) feedback->observe(stored.chat, self.user_id);
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
    state.arrival->set();
    // Due now, or once the idle backoff ends (the worker waits for it).
    if (!should_plan(state.flow, flow_settings(), now()) && !next_check(state.flow, flow_settings(), now())) return std::nullopt;
    return chat;
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
    if (const auto chat = ingest(*message)) wake(*chat);
}

void Responder::wake(const store::ChatKey &chat) {
    auto &state = chats[chat];
    if (state.busy) return; // the running worker looks again after its turn
    state.busy = true;
    spawn(turn_worker(this, chat));
}

} // namespace lilia::chat
