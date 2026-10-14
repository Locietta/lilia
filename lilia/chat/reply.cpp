#include <lilia/chat/detail/responder.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

#include <lighter/async/io/watcher.h>

#include <xmake/bin2obj/facts_md.hpp>
#include <xmake/bin2obj/select_md.hpp>
#include <xmake/bin2obj/self_facts_md.hpp>
#include <xmake/bin2obj/writer_group_md.hpp>
#include <xmake/bin2obj/writer_private_md.hpp>

#include <lilia/chat/delivery.h>
#include <lilia/chat/typos.h>
#include <lilia/log.h>

// One reply: the writer's text, sent the way MaiBot sends it; expression
// selection before it and the facts learned after it.

namespace lilia::chat {

using lighter::Task;
using namespace detail;

namespace {

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

/// Facts: evidence is the person's last few messages (at least this many
/// characters) plus the last lines of chat.
constexpr usize k_fact_own_messages = 3;
constexpr usize k_min_fact_evidence = 6;
constexpr usize k_fact_context = 8;

/// MaiBot's response_splitter: more pieces than this go out as one message;
/// the rest are packed into at most this many messages.
constexpr usize k_max_pieces = 8;
constexpr usize k_max_messages = 3;

std::string speaker(const Responder *r, const store::StoredMessage &message) {
    if (message.from_self) return r->config.chat.name + "（她自己）";
    return message.sender_name.empty() ? std::to_string(message.sender_id) : message.sender_name;
}

} // namespace

namespace detail {

/// MaiBot's expression selection, after the planner (or beside it when she is
/// called, without its intent): candidates retrieved by embedding against the
/// intent and the chat (MMR for variety), or a weighted sample without
/// embeddings; then a selector call with the recent chat picks 0 to
/// `max_selected`. Picks are marked active.
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
        // MaiBot's expression retrieval: by meaning, MMR for variety.
        std::vector<retrieval::Document> documents;
        for (const auto &row : *rows) documents.push_back({.id = row.id, .texts = {learn::expression_text(row.situation, row.style)}});
        std::string query = std::format("{}\n要回应的消息：{}\n最近的聊天：\n", intent, target.text);
        for (usize i = recent.size() > 3 ? recent.size() - 3 : 0; i < recent.size(); ++i) query += recent[i] + "\n";
        const auto matches = co_await r->memory->retriever.search(retrieval::k_expression_task, std::move(query), documents,
                                                                  {.top_k = learning.candidates, .mmr_lambda = k_mmr_lambda});
        for (const auto &match : matches) candidates.push_back((*rows)[match.index]);
        if (candidates.size() < k_min_candidates) candidates.clear();
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
    const auto items = co_await call_selector(r, chat, request.history, request.tools);
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

} // namespace detail

namespace {

/// What she knows about the person she answers, then about anyone else the
/// planner looked up this turn, for the writer.
std::vector<KnownPerson> known_people(Responder *r, const store::StoredMessage &target,
                                      const std::vector<std::pair<i64, KnownPerson>> &looked_up) {
    std::vector<KnownPerson> known;
    if (!r->memory || !r->config.memory.facts) return known;
    if (!target.from_self && !is_notice(target)) {
        auto facts = r->store.facts(target.sender_id, r->config.memory.facts_per_person);
        if (facts.has_value() && !facts->empty()) known.push_back({.name = display_name(target), .facts = memory::render_facts(*facts)});
    }
    for (const auto &[user, person] : looked_up) {
        if (user != target.sender_id || target.from_self) known.push_back(person);
    }
    return known;
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

namespace detail {

Task<Delivered> deliver(Responder *r, store::ChatKey chat, Plan plan, std::vector<store::StoredMessage> log_lines,
                        store::StoredMessage target, ChatInfo info, i64 started, Findings found) {
    // She types while the selector and the writer work (see the sending loop).
    const auto decided = std::chrono::steady_clock::now();
    auto &state = r->chats[chat];
    const auto &settings = r->config.chat;
    std::vector<std::string> habits;
    if (found.habits && found.habits_for == plan.target) {
        habits = *std::move(found.habits);
    } else {
        habits = co_await select_habits(r, chat, log_lines, target, plan.intent.empty() ? plan.brief : plan.intent);
    }
    const auto previous = std::ranges::find(state.replied, plan.target, &std::pair<i64, std::string>::first);
    const auto history = build_writer_history({
        .persona = r->persona,
        .rules = situated(r->writer_rules, chat, {}, embedded(xmake::writer_group_md), embedded(xmake::writer_private_md)),
        .info = info,
        .recent = log_lines,
        .target = target,
        .plan = plan,
        .now = started,
        .utc_offset_minutes = settings.utc_offset_minutes,
        .log_bytes = std::numeric_limits<usize>::max(), // log_window() bounds it
        .reference = std::move(found.online),
        .habits = habits,
        .remembered = std::move(found.remembered),
        .known = known_people(r, target, found.people),
        .self_notes = std::move(found.self_notes),
        .presence = std::move(found.presence),
        .previous_reply = previous == state.replied.end() ? std::string() : previous->second,
    });
    // Emoji go unless people use them (they show up in learned expressions).
    auto styles = r->store.expression_styles();
    const ReplyLimits limits{
        .max_bytes = settings.max_reply_bytes,
        .max_code_points = plan.style == ReplyStyle::SHORT  ? settings.short_reply_chars :
                           plan.style == ReplyStyle::NORMAL ? settings.normal_reply_chars :
                                                              settings.long_reply_chars,
        .kept_emoji = emoji_in(styles.has_value() ? *styles : std::vector<std::string>{}),
    };
    // The writer, then its fallbacks while nothing usable comes back. Every
    // repair is logged by kind (never the text) so their rates can be read.
    Repaired repaired;
    for (usize attempt = 0; attempt <= r->writer_fallbacks.size() && repaired.text.empty(); ++attempt) {
        auto &provider = attempt == 0 ? r->writer : r->writer_fallbacks[attempt - 1].provider;
        const auto &model = attempt == 0 ? r->config.model.model : r->writer_fallbacks[attempt - 1].model;
        if (attempt > 0) log::warn("writer fallback for {}: {}", describe(chat), model);
        const auto written = co_await call_model(r, provider, model, "writer", chat, history, {});
        if (!written) continue;
        const auto raw = final_text(*written);
        log::debug("writer raw: {}", raw);
        repaired = repair_reply(raw, log_lines, info, settings.aliases, limits);
        if (!repaired.repairs.empty()) {
            std::string kinds;
            for (const auto kind : repaired.repairs) kinds += std::format("{}{}", kinds.empty() ? "" : ", ", kind);
            log::warn("{} writer output repaired ({}): {}", describe(chat), model, kinds);
        }
    }
    const auto &reply = repaired.text;

    // Casual replies go out the way MaiBot sends them: split at punctuation,
    // neighbours merged at random, a homophone typo now and then (sometimes
    // corrected right after), at most three messages. A long reply is on
    // purpose (an explanation, a copypasta): one message, untouched.
    std::vector<Segment> segments;
    if (plan.style == ReplyStyle::LONG) {
        for (auto &whole : whole_message(reply)) segments.push_back({.text = std::move(whole)});
    } else {
        // Every line, trimmed, blank ones dropped; no spaces around English
        // words, which the split would take for breaks.
        const auto casual = tighten_spacing(first_lines(reply, std::numeric_limits<usize>::max()));

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
        log::warn("writer returned an empty reply for {}", describe(chat));
        co_return Delivered{};
    }
    if (r->observer) r->observer(chat, plan, bubbles);

    std::optional<i64> sent_id;
    usize sent = 0;
    std::vector<store::StoredMessage> own; ///< what went out, for her own facts
    for (usize i = 0; i < segments.size(); ++i) {
        // Each message waits for its "typing". The first was being typed since
        // the planner decided, so a short one goes out as soon as it is
        // written; a deliberate long reply's first goes out at once.
        if (r->realtime && (i > 0 || plan.style != ReplyStyle::LONG)) {
            auto typing = typing_delay(segments[i].text, settings.typing_speed);
            if (i == 0) typing -= std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decided);
            if (typing.count() > 0) co_await lighter::sleep(typing);
        }
        onebot::Message message;
        if (i == 0 && plan.quote && chat.type == "group" && plan.target > 0) {
            message.push_back(onebot::reply_segment(plan.target));
        } else if (segments[i].quote_previous && sent_id) {
            message.push_back(onebot::reply_segment(*sent_id));
        }
        message.push_back(onebot::text_segment(segments[i].text));
        sent_id = co_await send_and_record(r, chat, std::move(message));
        if (!sent_id) break;
        if (i == 0) {
            log::info("{} first message out {} ms after the decision", describe(chat),
                      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decided).count());
        }
        own.push_back({.chat = chat, .message_id = *sent_id, .time = r->now(), .text = segments[i].text, .from_self = true});
        ++sent;
    }
    if (sent == 0) co_return Delivered{};
    if (r->memory && r->config.memory.facts) r->spawn(learn_self_facts(r, chat, own, log_lines));
    note_spoke(state.flow);
    if (r->memory && r->config.memory.facts && !target.from_self && !is_notice(target)) r->spawn(learn_facts(r, chat, target, log_lines));

    std::string said;
    for (usize i = 0; i < sent; ++i) said += (said.empty() ? "" : " ") + bubbles[i];
    state.replied.emplace_back(plan.target, said);
    if (state.replied.size() > k_replied_memory) state.replied.pop_front();
    std::vector<i64> ids;
    for (const auto &message : own) ids.push_back(message.message_id);
    co_return Delivered{.sent = true, .said = std::move(said), .message_ids = std::move(ids)};
}

} // namespace detail

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

} // namespace lilia::chat
