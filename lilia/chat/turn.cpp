#include <lilia/chat/detail/responder.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <variant>

#include <glaze/json.hpp>

#include <lighter/async/io/watcher.h>
#include <lighter/async/runtime/timeout.h>
#include <lighter/async/runtime/when.h>

#include <xmake/bin2obj/planner_group_md.hpp>
#include <xmake/bin2obj/planner_md.hpp>
#include <xmake/bin2obj/planner_private_md.hpp>

#include <lilia/chat/delivery.h>
#include <lilia/learn/frequency.h>
#include <lilia/log.h>
#include <lilia/json.h>

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

/// MaiBot's planner loop: rounds per turn (its 16 since 074a8f41) and waits
/// in a row (its limits); replies per turn is lilia's own brake.
constexpr usize k_max_rounds = 16;
constexpr usize k_max_waits = 3;
constexpr usize k_max_replies = 3;

/// Web searches, people looked up and sticker searches per turn.
constexpr usize k_max_web_searches = 2;
constexpr usize k_max_look_ups = 3;
constexpr usize k_max_sticker_finds = 3;
/// What the planner is told about the addressee hint, when hints are on.
constexpr std::string_view k_hint_rule =
    "- 「像是在对她说」的估计是根据这个群以往的情况算出来的，只是参考；和聊天内容不一致时，以聊天内容为准。\n";
/// A round's analysis as logged in planner_turns.
constexpr usize k_logged_analysis_bytes = 400;

/// At most `limit` bytes of `text`, cut at a code point boundary.
std::string first_bytes(std::string_view text, usize limit) {
    if (text.size() <= limit) return std::string(text);
    usize end = limit;
    while (end > 0 && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) --end;
    return std::string(text.substr(0, end));
}

/// A message that is a single sticker, ready to send again (复读).
std::optional<onebot::Segment> sticker_again(const store::StoredMessage &message) {
    onebot::Message segments;
    if (glz::read<onebot::k_read_opts>(segments, message.segments_json)) return std::nullopt;
    return sticker::resend_segment(segments);
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
    auto frequent = r->store.frequent_terms(chat);
    if (!r->config.learning.jargon_groups.empty()) {
        std::erase_if(*terms, [&](const store::Jargon &term) {
            if (term.created_by == "MANUAL") return false;
            if (frequent.has_value()) {
                const auto norm_term = learn::normalize_match_text(term.content);
                if (std::ranges::any_of(*frequent,
                                        [&](const store::FrequentTerm &ft) { return learn::normalize_match_text(ft.term) == norm_term; })) {
                    return false;
                }
            }
            auto chats = r->store.jargon_chats(term.id);
            return chats.has_value() &&
                   std::ranges::none_of(*chats, [&](i64 id) { return learn::shares(r->config.learning.jargon_groups, chat.id, id); });
        });
    }
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

/// The planner's analysis: the text it wrote before its tool calls.
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

/// What comes after a tool call: the round's next call, the next round (its
/// later calls are not run), or the end of the turn.
enum struct Next {
    GO_ON,
    PAUSE,
    END,
};

/// What one tool call came to.
struct Step {
    std::string receipt; ///< its result, for the planner
    Next next = Next::GO_ON;
    std::string ended; ///< why the turn ended (Next::END)
};

/// One planner tool: whether the chat offers it this turn, how often a turn
/// may call it, and what a call does.
struct ToolHandler {
    std::string_view name;
    bool available = true;
    usize limit = std::numeric_limits<usize>::max(); ///< calls per turn
    bool stops = false;                              ///< the turn may stop here (wait, stay_silent)
    /// A call, its arguments as read and its planner_turns row.
    std::move_only_function<Task<Step>(const provider::ToolCall &, Plan, i64)> run;
    usize used = 0; ///< calls run this turn
};

} // namespace

Task<> Responder::run_turn(store::ChatKey chat) {
    auto &state = chats[chat];
    auto fresh = std::exchange(state.flow.unseen, {});
    auto addressed = std::exchange(state.flow.addressed, std::nullopt);
    auto named = std::exchange(state.flow.named, std::nullopt);
    if (fresh.empty() || state.muted_until > now()) co_return;

    const auto &settings = config.chat;
    const i64 started = now();
    const auto opened = std::chrono::steady_clock::now();
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
    const ChatInfo info{.chat = chat, .chat_name = state.name, .name = settings.name, .self_name = self_name(chat)};
    const bool knows_people = memory && config.memory.facts;
    const bool has_stickers = stickers_ready(this);
    const auto rule = memory_rule(memory != nullptr, knows_people);
    const auto prompt = situated(planner_prompt.empty() ? embedded(xmake::planner_md) : std::string_view(planner_prompt), chat,
                                 {{"name", settings.name},
                                  {"hint_rule", feedback && config.feedback.hint ? std::string_view(k_hint_rule) : std::string_view()},
                                  {"persona", persona},
                                  {"memory_rule", rule},
                                  {"search_rule", search_rule(config.model.web_search)},
                                  {"sticker_rule", sticker_rule(has_stickers)}},
                                 embedded(xmake::planner_group_md), embedded(xmake::planner_private_md));
    const PlannerToolSet offered{.memory = memory != nullptr,
                                 .people = knows_people,
                                 .web = config.model.web_search,
                                 .stickers = has_stickers,
                                 .poke = chat.type == "group"};
    const auto tools = planner_tools(offered);
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
    log::info("{} ready to plan in {} ms", describe(chat),
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - opened).count());
    // What the turn finds out reaches the writer as it is, not through the
    // planner's words.
    Findings found{.remembered = notes.recall,
                   .earlier = memory ? memory->earlier(chat, 1) : std::string(),
                   .self_notes = notes.self_notes,
                   .presence = notes.presence,
                   .jargon = notes.jargon};

    const auto row_of = [&rows](i64 message_id) -> i64 {
        const auto it = std::ranges::find(rows, message_id, &store::StoredMessage::message_id);
        return it == rows.end() ? 0 : it->row_id;
    };
    // Whom each line is said to, from its @s, quotes and her name; computed
    // once per row, so the rendered history stays a cacheable prefix.
    const auto edges_of = [&](const std::vector<store::StoredMessage> &lines) {
        const auto lookup = [&](i64 message_id) -> std::optional<store::StoredMessage> {
            if (const auto it = std::ranges::find(lines, message_id, &store::StoredMessage::message_id); it != lines.end()) return *it;
            auto stored = store.message(chat, message_id);
            return stored.has_value() && *stored ? std::optional<store::StoredMessage>(**stored) : std::nullopt;
        };
        const auto name_of = [&](i64 user) { return person(this, chat, user); };
        std::vector<LineEdges> out;
        for (const auto &line : lines) {
            auto it = state.edges.find(line.row_id);
            if (it == state.edges.end()) {
                it = state.edges.emplace(line.row_id, line_edges(line, self.user_id, info.self_name, settings.aliases, lookup, name_of))
                         .first;
            }
            out.push_back(it->second);
        }
        // Rows before the window are not shown again.
        if (!lines.empty()) std::erase_if(state.edges, [&](const auto &entry) { return entry.first < lines.front().row_id; });
        return out;
    };
    // Facts about the lines new to this round, not instructions: the planner decides.
    const auto facts = [&](const std::optional<i64> &called, i64 from_row, const std::vector<LineEdges> &edges) {
        std::vector<std::string> out;
        if (chat.type != "group") return out;
        if (const auto it = called ? std::ranges::find(rows, *called, &store::StoredMessage::message_id) : rows.end();
            it != rows.end() && is_notice(*it)) {
            out.push_back(std::format("#{} 有人戳了戳她。", it->row_id));
        }
        const auto join = [](const std::vector<std::string> &items) {
            std::string joined;
            for (const auto &item : items) joined += (joined.empty() ? "" : "、") + item;
            return joined;
        };
        // Whom a line @s or quotes is on the line itself (→她, →小明); her
        // name in the text is not, and may be talk about her.
        std::vector<std::string> named;
        for (usize i = 0; from_row > 0 && i < rows.size() && i < edges.size(); ++i) {
            if (rows[i].row_id < from_row || rows[i].from_self || is_notice(rows[i])) continue;
            if (!edges[i].to_her && edges[i].names_her) named.push_back(std::format("#{}", rows[i].row_id));
        }
        if (!named.empty()) out.push_back(std::format("{} 提到了她的名字。", join(named)));
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
        if (auto latest = log_window(chat); !latest.empty()) {
            rows = std::move(latest);
            notes.jargon = jargon_notes(this, chat, rows);
            found.jargon = notes.jargon;
        }
        if (!arrived.empty()) newest = arrived.back().message_id;
    };
    const auto reply_to = [&](Plan plan) -> Task<Delivered> {
        const auto target = std::ranges::find(rows, plan.target, &store::StoredMessage::message_id);
        // She answers others, not herself: the planner picks again.
        if (target != rows.end() && target->from_self)
            co_return Delivered{.refused = std::format("#{} 是她自己说的，选别人的一条。", target->row_id)};
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
        log::info("{} replies to {} ({}, to her {}, quote {}): {}", describe(chat), plan.target, style_name(plan.style),
                  plan.to_her.empty() ? "?" : plan.to_her, plan.quote, plan.intent);
        co_return co_await deliver(this, chat, plan, rows, *target, info, now(), found);
    };

    usize waits = 0;
    usize replies = 0;
    std::string analysis;
    // Replay only (`--continue-at`, blind comparisons): she must say something
    // this turn, so neither a tool that stops nor a round without calls ends
    // it before she has.
    const auto must_still_speak = [&] { return must_speak && replies == 0; };
    // Each of her messages counts against the turn's replies.
    const auto spoke = [&](std::string receipt) {
        ++replies;
        waits = 0;
        note_spoke(state.flow);
        if (replies >= k_max_replies) return Step{.receipt = std::move(receipt), .next = Next::END, .ended = "replied enough"};
        return Step{.receipt = std::move(receipt)};
    };
    // Whoever wrote the line, or whoever a notice says did it (poked her); 0
    // for her own lines and lines not shown.
    const auto author_of = [&](i64 message_id) -> i64 {
        const auto target = std::ranges::find(rows, message_id, &store::StoredMessage::message_id);
        if (target == rows.end() || target->from_self) return 0;
        return is_notice(*target) ? notice_actor(*target) : target->sender_id;
    };

    // The planner's tools, in the order planner_tools() offers them.
    std::array handlers{
        ToolHandler{
            .name = "reply",
            .run = [&](const provider::ToolCall &, Plan plan, i64 turn_id) -> Task<Step> {
                const auto delivered = co_await reply_to(plan);
                if (!delivered.refused.empty()) co_return Step{.receipt = delivered.refused};
                if (!delivered.sent) co_return Step{.receipt = "没发出去。", .next = Next::END, .ended = "the reply failed"};
                if (turn_id > 0) {
                    std::string ids;
                    for (const auto id : delivered.message_ids) ids += (ids.empty() ? "" : ",") + std::to_string(id);
                    std::ignore = store.set_planner_turn_sent(turn_id, ids);
                }
                co_return spoke(std::format("她回复了 #{}，说的话在上面的聊天记录里。", row_of(plan.target)));
            },
        },
        ToolHandler{
            // Stickers she could send: the planner picks one (or none) next round.
            .name = "find_sticker",
            .available = offered.stickers,
            .limit = k_max_sticker_finds,
            .run = [&](const provider::ToolCall &call, Plan, i64) -> Task<Step> {
                const auto feeling = string_field(call.input, "feeling");
                const auto candidates = co_await find_stickers(this, feeling);
                log::info("{} looks for stickers for \"{}\": {} found", describe(chat), feeling, candidates.size());
                co_return Step{.receipt = candidates.empty() ? std::string("没找到能发的表情包。") :
                                                               std::format("找到这些：\n{}要发就用 send_sticker 填编号；都不贴切就别发。",
                                                                           sticker::render_candidates(candidates, now()))};
            },
        },
        ToolHandler{
            // Any registered sticker by its number, usually one find_sticker showed.
            .name = "send_sticker",
            .available = offered.stickers,
            .run = [&](const provider::ToolCall &, Plan plan, i64) -> Task<Step> {
                if (now() - state.last_sticker < config.stickers.cooldown_seconds)
                    co_return Step{.receipt = "她刚发过表情包，隔一会儿再发；可以用文字回应，或者不说话。"};
                std::optional<store::Sticker> chosen;
                if (const auto id = sticker::parse_handle(plan.sticker)) {
                    if (auto pool = store.stickers("registered"); pool.has_value()) {
                        if (const auto it = std::ranges::find(*pool, *id, &store::Sticker::id); it != pool->end()) chosen = *it;
                    }
                }
                if (!chosen) co_return Step{.receipt = "没有这个编号的表情包；编号要用 find_sticker 找到的，比如 s12。"};
                if (!co_await send_sticker(this, chat, *chosen)) co_return Step{.receipt = "没发出去。"};
                if (observer) observer(chat, plan, {"[表情]"});
                co_return spoke(std::format("她发了表情包 {}。", sticker::handle(*chosen)));
            },
        },
        ToolHandler{
            // 复读: the message again, as it is (a sticker too).
            .name = "repeat",
            .run = [&](const provider::ToolCall &, Plan plan, i64) -> Task<Step> {
                const auto target = std::ranges::find(rows, plan.target, &store::StoredMessage::message_id);
                const auto again = target != rows.end() && !target->from_self ? sticker_again(*target) : std::nullopt;
                if (target == rows.end() || target->from_self || is_notice(*target) || (!again && !repeatable(target->text))) {
                    co_return Step{.receipt = "这条不适合复读（只复读别人的短消息或表情包），没有发。"};
                }
                if (!co_await send_and_record(this, chat,
                                              again ? onebot::Message{*again} : onebot::Message{onebot::text_segment(target->text)},
                                              again ? target->text : std::string())) {
                    co_return Step{.receipt = "没发出去。"};
                }
                if (observer) observer(chat, plan, {target->text});
                log::info("{} repeats {}: {}", describe(chat), plan.target, target->text);
                co_return spoke(std::format("她复读了 #{}。", target->row_id));
            },
        },
        ToolHandler{
            .name = "wait",
            .stops = true,
            .run = [&](const provider::ToolCall &, Plan plan, i64) -> Task<Step> {
                // MaiBot allows three waits in a row, then the turn rests.
                if (++waits > k_max_waits)
                    co_return Step{.receipt = "不能再等了，这一轮到此为止。", .next = Next::END, .ended = "waited enough"};
                log::info("{} waits {}s: {}", describe(chat), plan.wait, plan.reason);
                if (observer) observer(chat, plan, {});
                const i64 before = now();
                if (pause) {
                    co_await pause(chat, plan.wait);
                } else if (realtime && chat.type == "private") {
                    // MaiBot's: in a private chat, the other one writing ends the wait.
                    auto arrival = state.arrival;
                    arrival->reset();
                    if (state.flow.unseen.empty())
                        std::ignore = co_await lighter::with_timeout(arrival->wait(), std::chrono::seconds(plan.wait));
                } else if (realtime) {
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(plan.wait);
                    auto arrival = state.arrival;
                    while (!state.flow.addressed) {
                        arrival->reset();
                        if (state.flow.addressed) break;
                        const auto remaining =
                            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
                        if (remaining <= std::chrono::milliseconds::zero()) break;
                        auto arrived = co_await lighter::with_timeout(arrival->wait(), remaining);
                        if (!arrived.has_value()) break;
                    }
                }
                std::string receipt;
                if (chat.type == "private" && !state.flow.unseen.empty()) {
                    receipt = std::format("等了 {} 秒，对方发来了新消息。", now() - before);
                } else if (chat.type == "group" && state.flow.addressed) {
                    receipt = std::format("等了 {} 秒，有人叫了她。", now() - before);
                } else {
                    receipt = std::format("等了 {} 秒。", now() - before);
                }
                co_return Step{.receipt = std::move(receipt), .next = Next::PAUSE};
            },
        },
        ToolHandler{
            .name = "poke",
            .available = offered.poke,
            .run = [&](const provider::ToolCall &, Plan plan, i64) -> Task<Step> {
                const i64 user = author_of(plan.target);
                if (user == 0 || user == self.user_id) co_return Step{.receipt = "这条找不到可以戳的人。"};
                if (now() - state.poked[user] < k_poke_gap_seconds) co_return Step{.receipt = "刚戳过这个人，过一会儿再戳。"};
                if (!co_await call_action(this, "send_poke", std::format(R"({{"group_id":{},"user_id":{}}})", chat.id, user))) {
                    co_return Step{.receipt = "没戳成。"};
                }
                state.poked[user] = now();
                note_spoke(state.flow);
                if (observer) observer(chat, plan, {"[戳一戳]"});
                log::info("{} pokes {}", describe(chat), user);
                co_return Step{.receipt = std::format("她戳了戳 {}。", person(this, chat, user))};
            },
        },
        ToolHandler{
            // A look into the history: its result goes back to the planner,
            // and to the writer if she replies.
            .name = "search_history",
            .available = offered.memory,
            .limit = config.memory.max_searches,
            .run = [&](const provider::ToolCall &call, Plan, i64) -> Task<Step> {
                const auto hits = co_await memory->search(chat, string_field(call.input, "query"));
                const auto rendered = memory::render_hits(hits);
                if (!hits.empty()) found.remembered += rendered;
                co_return Step{.receipt = rendered};
            },
        },
        ToolHandler{
            // MaiBot's person profiles, on request: whoever wrote the line.
            .name = "look_up_person",
            .available = offered.people,
            .limit = k_max_look_ups,
            .run = [&](const provider::ToolCall &, Plan plan, i64) -> Task<Step> {
                const i64 user = author_of(plan.target);
                if (user == 0) co_return Step{.receipt = "这条找不到是谁发的。"};
                if (!memory) co_return Step{.receipt = "现在用不了 look_up_person。"};
                const auto name = person(this, chat, user);
                std::string context;
                for (auto it = rows.end() - static_cast<std::ptrdiff_t>(std::min<usize>(rows.size(), k_self_context)); it != rows.end();
                     ++it)
                    context += it->text + "\n";
                const auto rendered = co_await memory->person_notes(user, chat, std::move(context), config.memory.facts_per_person);
                if (rendered.empty()) co_return Step{.receipt = std::format("她对{}还没记下什么。", name)};
                if (!std::ranges::contains(found.people, user, &std::pair<i64, KnownPerson>::first))
                    found.people.push_back({user, {.name = name, .facts = rendered}});
                co_return Step{.receipt = std::format("她记得的关于{}的事：{}", name, rendered)};
            },
        },
        ToolHandler{
            // A look online: the same as search_history.
            .name = "search_web",
            .available = offered.web,
            .limit = k_max_web_searches,
            .run = [&](const provider::ToolCall &call, Plan, i64) -> Task<Step> {
                const auto online = co_await search_web(this, chat, info, string_field(call.input, "query"));
                if (!online.empty()) found.online += (found.online.empty() ? "" : "\n") + online;
                co_return Step{.receipt = online.empty() ? std::string("没查到。") : "查到的：" + online};
            },
        },
        ToolHandler{
            .name = "stay_silent",
            .stops = true,
            .run = [&](const provider::ToolCall &, Plan plan, i64) -> Task<Step> {
                co_return Step{.receipt = "好。", .next = Next::END, .ended = plan.reason};
            },
        },
    };
    // One tool call of a round, with its arguments as read and its
    // planner_turns row. A mistake gets a receipt and the round goes on, so the
    // planner sees it; the tools never change, which would void the cached
    // prompt.
    const auto act = [&](const provider::ToolCall &call, std::expected<Plan, std::string> parsed, i64 turn_id) -> Task<Step> {
        const auto handler = std::ranges::find(handlers, std::string_view(call.name), &ToolHandler::name);
        if (handler == handlers.end()) co_return Step{.receipt = std::format("没有 {} 这个工具。", call.name)};
        if (!handler->available || handler->limit == 0) co_return Step{.receipt = std::format("现在用不了 {}。", call.name)};
        if (handler->used >= handler->limit)
            co_return Step{.receipt = std::format("这次已经用过 {} 次 {} 了，根据现有的信息决定。", handler->used, call.name)};
        if (handler->stops && must_still_speak()) co_return Step{.receipt = "这次需要她说点什么，用 reply。"};
        if (!parsed) co_return Step{.receipt = std::move(parsed).error()};
        ++handler->used;
        parsed->follow_up = replies > 0;
        parsed->analysis = analysis;
        co_return co_await handler->run(call, *std::move(parsed), turn_id);
    };

    // MaiBot's planner loop: each round an analysis and tool calls, run in
    // order; the round, its results and the lines that came in meanwhile stay
    // in the chat's history for every later round and turn.
    i64 seen_row = 0;
    std::string ended = "planner call failed";
    bool over = false;
    for (usize round = 0; round < k_max_rounds && !over; ++round) {
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
        const auto edges = edges_of(rows);
        tail.facts = facts(addressed, tail.first_new_row, edges);
        tail.now = now();
        // The addressee prior: shown as a hint when on, always logged.
        const auto hint =
            feedback && chat.type == "group" ? feedback->hints(chat, rows, edges, tail.first_new_row) : feedback::Feedback::Hint{};
        tail.hint = hint.shown;
        const i64 shown = rows.back().row_id;
        const auto earlier = memory ? memory->earlier(chat) : std::string();
        const auto pictures = pictures_of(store, config.data_dir, rows, picture_cache);
        const auto rendered = render_session(prompt, info, rows, state.records, tail, edges, earlier, pictures);
        // Called, she will most likely answer the line that called her: its
        // expressions are picked while the planner decides, not after it.
        const auto called = addressed ? addressed : named;
        const auto calling = called ? std::ranges::find(rows, *called, &store::StoredMessage::message_id) : rows.end();
        std::optional<std::vector<provider::OutputItem>> planned;
        if (calling != rows.end() && !calling->from_self && !is_notice(*calling)) {
            auto [decided, picked] = co_await lighter::when_all(call_planner(this, "planner", chat, rendered, tools),
                                                                select_habits(this, chat, rows, *calling, {}));
            planned = std::move(decided);
            found.habits_for = calling->message_id;
            found.habits = std::move(picked);
        } else {
            planned = co_await call_planner(this, "planner", chat, rendered, tools);
        }
        if (!planned) break;
        keep_warm(this, chat, rendered, tools);
        seen_row = shown;
        addressed.reset();
        named.reset();
        analysis = analysis_of(*planned);
        if (!analysis.empty()) log::info("{} planner: {}", describe(chat), analysis);
        std::vector<const provider::ToolCall *> calls;
        for (const auto &item : *planned) {
            if (const auto *call = std::get_if<provider::ToolCallItem>(&item)) calls.push_back(&call->call);
        }

        // Every call goes on record (planner_turns): what the round saw and did.
        const auto log_turn = [&](std::string tool, std::string what, const Plan &plan) {
            auto logged = store.add_planner_turn({.time = now(),
                                                  .chat = chat,
                                                  .cycle = cycle,
                                                  .round = static_cast<i64>(round),
                                                  .first_new_row = tail.first_new_row,
                                                  .shown_row = shown,
                                                  .tool = std::move(tool),
                                                  .argument = std::move(what),
                                                  .target_message_id = plan.target,
                                                  .to_her = plan.to_her,
                                                  .analysis = first_bytes(analysis, k_logged_analysis_bytes),
                                                  .reason = plan.reason,
                                                  .hint = hint.logged});
            return logged.has_value() ? *logged : i64{0};
        };
        // The round stays in the history with a result for each call.
        PlannerRecord record{.after_row = shown, .cycle = cycle, .output = *planned};
        if (calls.empty()) {
            // No call: she says nothing (replay's must_speak asks again).
            std::ignore = log_turn({}, {}, {.reason = "no tool call"});
            state.records.push_back(std::move(record));
            if (must_still_speak()) {
                co_await refresh();
                continue;
            }
            ended = analysis.empty() ? std::string("no tool call") : analysis;
            break;
        }
        std::string skipped; // why the rest of the round's calls are not run
        for (const auto *call : calls) {
            if (!skipped.empty()) {
                record.outcomes.push_back({.call_id = call->id, .receipt = skipped});
                continue;
            }
            auto parsed = parse_call(*call, rows);
            auto what = string_field(call->input, "feeling");
            if (what.empty()) what = string_field(call->input, "query");
            if (what.empty()) what = string_field(call->input, "sticker");
            const i64 turn_id = log_turn(call->name, std::move(what), parsed ? *parsed : Plan{.reason = parsed.error()});
            const auto step = co_await act(*call, std::move(parsed), turn_id);
            record.outcomes.push_back({.call_id = call->id, .receipt = step.receipt});
            if (step.next == Next::PAUSE) skipped = "没有执行：等完先看看情况，再决定要不要做。";
            if (step.next == Next::END) {
                skipped = "没有执行：这一轮已经结束了。";
                ended = step.ended;
                over = true;
            }
        }
        state.records.push_back(std::move(record));
        if (!over) co_await refresh();
    }

    // Replay's must_speak: a turn that ended without her saying anything
    // replies to the newest line.
    if (must_still_speak() && ended != "the reply failed") {
        log::info("{} turn ended ({}) without a reply; replying anyway", describe(chat), ended);
        if ((co_await reply_to({.reply = true, .target = newest})).sent) ++replies;
    }
    if (replies == 0) {
        log::info("{} stays silent: {}", describe(chat), ended);
        note_silent(state.flow, now());
        if (observer) observer(chat, {.reason = ended, .analysis = analysis}, {});
    } else {
        log::info("{} ends the turn after {} repl{}: {}", describe(chat), replies, replies == 1 ? "y" : "ies", ended);
        if (observer) observer(chat, {.follow_up = true, .reason = ended, .analysis = analysis}, {});
    }
}

} // namespace lilia::chat
