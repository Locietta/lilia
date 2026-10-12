#include <lilia/chat/detail/responder.h>

#include <algorithm>
#include <chrono>
#include <format>
#include <optional>
#include <utility>
#include <variant>

#include <glaze/json.hpp>

#include <lighter/async/io/watcher.h>
#include <lighter/async/runtime/when.h>

#include <xmake/bin2obj/planner_md.hpp>

#include <lilia/chat/delivery.h>
#include <lilia/log.h>

// A turn: the planner's rounds and what each tool call does.

namespace lilia::chat {

using lighter::Task;
using namespace detail;

namespace {

/// Her own facts: the closest this many to the last `k_self_context` lines
/// (plus the newest few).
constexpr usize k_self_facts = 6;
constexpr usize k_self_context = 4;

/// She pokes the same person at most once in this many seconds.
constexpr i64 k_poke_gap_seconds = 60;

/// MaiBot's planner loop: rounds per turn and waits in a row (its limits);
/// replies per turn is lilia's own brake.
constexpr usize k_max_rounds = 8;
constexpr usize k_max_waits = 3;
constexpr usize k_max_replies = 3;

/// Web searches, people looked up and sticker searches per turn.
constexpr usize k_max_web_searches = 2;
constexpr usize k_max_look_ups = 3;
constexpr usize k_max_sticker_finds = 3;

/// A message that is a single sticker, ready to send again (复读).
std::optional<onebot::Segment> sticker_again(const store::StoredMessage &message) {
    onebot::Message segments;
    if (glz::read<onebot::k_read_opts>(segments, message.segments_json)) return std::nullopt;
    return sticker::resend_segment(segments);
}

/// A string argument of a tool call; empty when missing.
std::string string_input(const glz::generic &input, std::string_view key) {
    if (!input.is_object()) return {};
    const auto &object = input.get_object();
    const auto it = object.find(key);
    return it != object.end() && it->second.is_string() ? it->second.get_string() : std::string();
}

std::string_view style_name(ReplyStyle style) {
    switch (style) {
        case ReplyStyle::SHORT: return "short";
        case ReplyStyle::NORMAL: return "normal";
        case ReplyStyle::LONG: return "long";
    }
    return "?";
}

/// Nothing to add, as a task (the other side of a when_all).
Task<std::string> no_text() { co_return std::string(); }

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
    const bool knows_people = memory && config.memory.facts;
    const bool has_stickers = stickers_ready(this);
    const auto rule = memory_rule(memory != nullptr, knows_people);
    const auto prompt = fill_template(embedded(xmake::planner_md), {{"name", settings.name},
                                                                    {"persona", persona},
                                                                    {"handoff", handoff_rule(settings.handoff == "reference")},
                                                                    {"memory_rule", rule},
                                                                    {"search_rule", search_rule(config.model.web_search)},
                                                                    {"sticker_rule", sticker_rule(has_stickers)}});
    const auto tools = planner_tools({.reference_handoff = settings.handoff == "reference",
                                      .memory = memory != nullptr,
                                      .people = knows_people,
                                      .web = config.model.web_search,
                                      .stickers = has_stickers,
                                      .poke = chat.type == "group"});
    const u64 cycle = ++state.cycles;

    // Notes for this turn: in the tail of every round, never in the history.
    PlannerTail notes{.presence = std::exchange(state.presence_note, {}), .utc_offset_minutes = settings.utc_offset_minutes};
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
        if (const auto it = called ? std::ranges::find(rows, *called, &store::StoredMessage::message_id) : rows.end(); it != rows.end()) {
            out.push_back(is_notice(*it) ? std::format("#{} 有人戳了戳她。", it->row_id) :
                                           std::format("#{} 是叫她的（@ 了她或者回复了她的消息）。", it->row_id));
        }
        if (const i64 row = mentioned ? row_of(*mentioned) : 0) out.push_back(std::format("#{} 提到了她的名字。", row));
        // QQ's 复读: what people are doing, for the planner to join or not.
        if (const auto chain = repeat_chain(rows); chain && chain->people >= 2)
            out.push_back(std::format("最后 {} 个人在复读「{}」。", chain->people, chain->text));
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
    usize look_ups = 0;
    usize sticker_finds = 0;
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
        // A string argument of the round's tool call.
        const auto argument = [&](std::string_view key) {
            for (const auto &item : *planned) {
                if (const auto *call = std::get_if<provider::ToolCallItem>(&item)) return string_input(call->call.input, key);
            }
            return std::string();
        };
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
            (tool == "search_web" && web_searches >= k_max_web_searches) || (tool == "look_up_person" && look_ups >= k_max_look_ups) ||
            (tool == "find_sticker" && sticker_finds >= k_max_sticker_finds)) {
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
            const auto hits = co_await memory->search(chat, argument("query"));
            const auto found = memory::render_hits(hits);
            if (!hits.empty()) remembered += found;
            keep(found);
            ++searches;
            continue;
        }
        // A look online: what it finds goes back to the planner, and to the
        // writer if she replies.
        if (config.model.web_search && tool == "search_web") {
            const auto found = co_await search_web(this, chat, info, argument("query"));
            if (!found.empty()) online += (online.empty() ? "" : "\n") + found;
            keep(found.empty() ? std::string("没查到。") : "查到的：" + found);
            ++web_searches;
            continue;
        }
        // Stickers she could send: the planner picks one (or none) next round.
        if (has_stickers && tool == "find_sticker") {
            const auto feeling = argument("feeling");
            const auto found = co_await find_stickers(this, feeling);
            log::info("{} looks for stickers for \"{}\": {} found", describe(chat), feeling, found.size());
            keep(found.empty() ? std::string("没找到能发的表情包。") :
                                 std::format("找到这些：\n{}要发就用 send_sticker 填编号；都不贴切就别发。",
                                             sticker::render_candidates(found, now(), config.stickers.usage_in_search)));
            ++sticker_finds;
            continue;
        }

        auto plan = parse_plan(*planned, rows);
        plan.follow_up = replies > 0;
        plan.analysis = analysis;
        if (plan.look_up && knows_people) {
            // MaiBot's person profiles, on request: whoever wrote the line.
            const auto target = std::ranges::find(rows, plan.target, &store::StoredMessage::message_id);
            const i64 user = target == rows.end() || target->from_self ? 0 : is_notice(*target) ? notice_actor(*target) : target->sender_id;
            std::string receipt = "这条找不到是谁发的。";
            if (user != 0) {
                const auto name = person(this, chat, user);
                auto facts = store.facts(user, config.memory.facts_per_person);
                receipt = facts.has_value() && !facts->empty() ? std::format("她记得的关于{}的事：{}", name, memory::render_facts(*facts)) :
                                                                 std::format("她对{}还没记下什么。", name);
            }
            keep(std::move(receipt));
            ++look_ups;
            continue;
        }
        if (!plan.sticker.empty()) {
            // Any registered sticker by its number, usually one find_sticker showed.
            std::optional<store::Sticker> chosen;
            if (const auto id = sticker::parse_handle(plan.sticker)) {
                if (auto pool = store.stickers("registered"); pool.has_value()) {
                    if (const auto it = std::ranges::find(*pool, *id, &store::Sticker::id); it != pool->end()) chosen = *it;
                }
            }
            std::string receipt;
            if (!chosen) {
                receipt = "没有这个编号的表情包；编号要用 find_sticker 找到的，比如 s12。";
            } else if (co_await send_sticker(this, chat, *chosen)) {
                ++replies;
                note_spoke(state.flow);
                if (observer) observer(chat, plan, {"[表情]"});
                receipt = std::format("她发了表情包 {}。", sticker::handle(*chosen));
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
        if (plan.repeat) {
            // 复读: the message again, as it is (a sticker too).
            const auto target = std::ranges::find(rows, plan.target, &store::StoredMessage::message_id);
            const auto again = target != rows.end() && !target->from_self ? sticker_again(*target) : std::nullopt;
            std::string receipt;
            if (target == rows.end() || target->from_self || is_notice(*target) || (!again && !repeatable(target->text))) {
                receipt = "这条不适合复读（只复读别人的短消息或表情包），没有发。";
            } else if (co_await send_and_record(this, chat,
                                                again ? onebot::Message{*again} : onebot::Message{onebot::text_segment(target->text)},
                                                again ? target->text : std::string())) {
                ++replies;
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
        if (plan.poke) {
            // Whoever wrote the line, or whoever a poke notice says poked her.
            const auto target = std::ranges::find(rows, plan.target, &store::StoredMessage::message_id);
            const i64 user = target == rows.end() ? 0 :
                             is_notice(*target)   ? notice_actor(*target) :
                             target->from_self    ? 0 :
                                                    target->sender_id;
            std::string receipt;
            if (user == 0 || user == self.user_id) {
                receipt = "这条找不到可以戳的人。";
            } else if (now() - state.poked[user] < k_poke_gap_seconds) {
                receipt = "刚戳过这个人，过一会儿再戳。";
            } else if (co_await call_action(this, "send_poke", std::format(R"({{"group_id":{},"user_id":{}}})", chat.id, user))) {
                state.poked[user] = now();
                note_spoke(state.flow);
                if (observer) observer(chat, plan, {"[戳一戳]"});
                log::info("{} pokes {}", describe(chat), user);
                receipt = std::format("她戳了戳 {}。", person(this, chat, user));
            } else {
                receipt = "没戳成。";
            }
            keep(receipt);
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
            if (tool == "reply" || tool == "repeat" || tool == "poke" || tool == "look_up_person") {
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
