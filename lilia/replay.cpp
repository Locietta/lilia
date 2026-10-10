#include "replay.h"

#include <chrono>
#include <cstdio>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#include <glaze/json.hpp>

#include <lighter/async/runtime/task_group.h>

#include <lilia/chat/responder.h>
#include <lilia/learn/learner.h>
#include <lilia/onebot/message.h>
#include <lilia/provider/anthropic.h>
#include <lilia/store/store.h>

namespace lilia {

using lighter::Task;

namespace {

constexpr i64 k_replay_group = 1;
constexpr i64 k_replay_self = 10'001;
/// Expressions per review call in `lilia expressions --review`.
constexpr usize k_review_batch = 20;

struct ReplayLine {
    i64 t = 0;
    i64 uid = 0;
    std::string name;
    std::string text;
};

void print(std::string_view line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

/// "@莉莉娅 在吗" -> [at(self), text(" 在吗")] when the name is lilia's.
onebot::Message to_segments(std::string_view text, const std::vector<std::string> &names) {
    onebot::Message message;
    if (text.starts_with('@')) {
        for (const auto &name : names) {
            if (!name.empty() && text.substr(1).starts_with(name)) {
                message.push_back(onebot::at_segment(k_replay_self));
                text.remove_prefix(1 + name.size());
                break;
            }
        }
    }
    if (!text.empty()) message.push_back(onebot::text_segment(std::string(text)));
    return message;
}

std::string_view style_name(chat::ReplyStyle style) {
    switch (style) {
        case chat::ReplyStyle::SHORT: return "short";
        case chat::ReplyStyle::NORMAL: return "normal";
        case chat::ReplyStyle::LONG: return "long";
    }
    return "?";
}

std::string clock(i64 t) { return std::format("{:02}:{:02}", t / 60, t % 60); }

} // namespace

namespace {

provider::Provider make_client(const ModelConfig &model, const std::string &base_url, const std::string &api_key, const std::string &name,
                               u32 max_tokens, bool require_tool = false, bool bearer = false, const std::string &thinking_off = {}) {
    return pro::make_proxy<provider::ProviderFacade, anthropic::Client>(anthropic::ClientOptions{
        .auth = provider::api_key_auth(api_key, bearer),
        .base_url = base_url,
        .model = name,
        .reasoning_effort = model.reasoning_effort,
        .disable_thinking = !model.thinking,
        .thinking_off = thinking_off.empty() ? "disabled" : thinking_off,
        .require_tool = require_tool,
        .max_tokens = max_tokens,
    });
}

} // namespace

provider::Provider make_planner(const ModelConfig &model) {
    const bool own = !model.planner_base_url.empty();
    return make_client(model, own ? model.planner_base_url : model.base_url, own ? model.planner_api_key : model.api_key,
                       model.planner_model, model.planner_max_tokens, model.planner_force_tool, own && model.planner_bearer,
                       own ? model.planner_thinking_off : std::string());
}

provider::Provider make_vision(const ModelConfig &model) {
    return make_client(model, model.base_url, model.api_key, model.vision_model, 400);
}

provider::Provider make_learner(const ModelConfig &model) {
    return make_client(model, model.base_url, model.api_key, model.planner_model, 1'000, true);
}

provider::Provider make_searcher(const ModelConfig &model) {
    return make_client(model, model.base_url, model.api_key, model.planner_model, 1'200);
}

provider::Provider make_writer(const ModelConfig &model) {
    return pro::make_proxy<provider::ProviderFacade, anthropic::Client>(anthropic::ClientOptions{
        .auth = provider::api_key_auth(model.writer_api_key.empty() ? model.api_key : model.writer_api_key,
                                       !model.writer_api_key.empty() && model.writer_bearer),
        .base_url = model.writer_base_url.empty() ? model.base_url : model.writer_base_url,
        .model = model.model,
        .reasoning_effort = model.reasoning_effort,
        .disable_thinking = !model.thinking,
        .thinking_off = model.writer_thinking_off.empty() ? "disabled" : model.writer_thinking_off,
        .temperature = model.writer_temperature,
        .max_tokens = model.max_tokens,
    });
}

std::vector<chat::Responder::NamedProvider> make_writer_fallbacks(const ModelConfig &model) {
    std::vector<chat::Responder::NamedProvider> fallbacks;
    for (const auto &fallback : model.writer_fallbacks) {
        fallbacks.push_back({
            .model = fallback.model,
            .provider = pro::make_proxy<provider::ProviderFacade, anthropic::Client>(anthropic::ClientOptions{
                .auth = provider::api_key_auth(fallback.api_key.empty() ? model.api_key : fallback.api_key,
                                               !fallback.api_key.empty() && fallback.bearer),
                .base_url = fallback.base_url.empty() ? model.base_url : fallback.base_url,
                .model = fallback.model,
                .reasoning_effort = model.reasoning_effort,
                .disable_thinking = !model.thinking,
                .thinking_off = fallback.thinking_off.empty() ? "disabled" : fallback.thinking_off,
                .temperature = model.writer_temperature,
                .max_tokens = model.max_tokens,
            }),
        });
    }
    return fallbacks;
}

namespace {

std::optional<std::vector<ReplayLine>> read_log(const std::filesystem::path &log) {
    std::ifstream input(log, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot read %s\n", log.string().c_str());
        return std::nullopt;
    }
    std::vector<ReplayLine> lines;
    std::string raw;
    for (usize number = 1; std::getline(input, raw); ++number) {
        if (raw.find_first_not_of(" \t\r") == std::string::npos) continue;
        ReplayLine line;
        if (auto ec = glz::read<glz::opts{.null_terminated = false, .error_on_unknown_keys = false}>(line, raw)) {
            std::fprintf(stderr, "%s:%zu: %s\n", log.string().c_str(), number, glz::format_error(ec, raw).c_str());
            return std::nullopt;
        }
        lines.push_back(std::move(line));
    }
    return lines;
}

} // namespace

Task<int> run_replay(Config config, std::filesystem::path log, std::optional<std::filesystem::path> persona,
                     std::optional<std::filesystem::path> habits, std::optional<std::filesystem::path> writer_prompt,
                     std::optional<std::filesystem::path> stickers, std::optional<usize> continue_at) {
    auto parsed = read_log(log);
    if (!parsed) co_return 1;
    auto lines = *std::move(parsed);

    config.chat.groups = {k_replay_group};
    config.chat.private_users.clear();
    config.chat.admins.clear();
    auto store = store::Store::open_memory();
    if (store.has_error()) {
        std::fprintf(stderr, "%s\n", store.error().message().c_str());
        co_return 1;
    }
    if (habits) {
        auto source = store::Store::open_readonly(*habits);
        if (source.has_error()) {
            std::fprintf(stderr, "%s\n", source.error().message().c_str());
            co_return 1;
        }
        auto rows = source->expressions(config.learning.max_expressions);
        if (rows.has_error()) {
            std::fprintf(stderr, "%s\n", rows.error().message().c_str());
            co_return 1;
        }
        for (const auto &row : *rows) {
            i64 id = 0;
            for (i64 i = 0; i < row.count; ++i) {
                if (auto added = store->learn_expression(row.chat, row.situation, row.style, row.last_active); added.has_value())
                    id = *added;
            }
            if (id != 0 && !row.vector.empty()) std::ignore = store->set_expression_vector(id, row.model, row.vector);
        }
        print(std::format("(载入 {} 条学来的说法)", rows->size()));
    }

    lighter::TaskGroup tasks(lighter::EventLoop::current());
    chat::Responder responder(config, *store, make_planner(config.model), make_searcher(config.model), make_writer(config.model),
                              persona.value_or(std::filesystem::path(config.data_dir) / "persona.md"),
                              [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    if (auto loaded = responder.reload_persona(); loaded.has_error()) {
        std::fprintf(stderr, "%s\n", loaded.error().message().c_str());
        co_return 1;
    }
    if (writer_prompt) {
        std::ifstream input(*writer_prompt, std::ios::binary);
        if (!input) {
            std::fprintf(stderr, "cannot read %s\n", writer_prompt->string().c_str());
            co_return 1;
        }
        responder.writer_rules.assign(std::istreambuf_iterator<char>(input), {});
    }
    responder.writer_fallbacks = make_writer_fallbacks(config.model);
    sticker::Collector collector(config.stickers, config.data_dir, *store, [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    if (stickers) {
        auto source = store::Store::open_readonly(*stickers);
        auto pool = source.has_value() ? source->stickers("registered") :
                                         Result<std::vector<store::Sticker>>(lighter::outcome_error(source.error()));
        if (pool.has_error()) {
            std::fprintf(stderr, "%s\n", pool.error().message().c_str());
            co_return 1;
        }
        for (auto sticker : *pool) {
            sticker.path = (stickers->parent_path() / sticker.path).string(); // absolute: read from the live data dir
            std::ignore = store->add_sticker(sticker);
        }
        responder.stickers = &collector;
        print(std::format("(载入 {} 个表情包)", pool->size()));
    }
    responder.self = {.user_id = k_replay_self, .nickname = config.chat.name};
    responder.realtime = false;

    const i64 base = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    i64 sim = 0;
    responder.now = [&sim, base] { return base + sim; };
    memory::Memory memory(config.memory, config.chat.name, config.chat.utc_offset_minutes, *store);
    if (config.memory.enabled) {
        memory.now = [&sim, base] { return base + sim; };
        memory.summarizer = make_learner(config.model);
        memory.summarizer_model = config.model.planner_model;
        responder.memory = &memory;
    }
    i64 next_id = 1'000'000;
    responder.outbox = [&next_id, &sim](const store::ChatKey &, onebot::Message message) -> Task<std::optional<i64>> {
        std::string quote;
        if (onebot::reply_target(message)) quote = "↩ ";
        const bool sticker = std::ranges::any_of(message, [](const onebot::Segment &segment) { return segment.type == "image"; });
        print(std::format("[{}]   {}: {}{}", clock(sim), "莉莉娅", quote,
                          sticker ? std::string("[发了一张表情包]") : onebot::render_plain(message)));
        co_return ++next_id;
    };
    responder.observer = [](const store::ChatKey &, const chat::Plan &plan, const std::vector<std::string> &) {
        if (plan.wait > 0) {
            print(std::format("        · 等 {} 秒：{}", plan.wait, plan.reason));
        } else if (!plan.reply) {
            print(std::format("        · {}：{}", plan.follow_up ? "到此为止" : "不说话", plan.reason));
        } else {
            print(std::format("        · 回复 {}（{}{}{}）：{}", plan.target, style_name(plan.style), plan.quote ? "，引用" : "",
                              plan.search ? "，搜索" : "", plan.intent.empty() ? plan.brief : plan.intent));
        }
    };

    std::vector<std::string> names = config.chat.aliases;
    names.push_back(config.chat.name);
    i64 message_id = 0;
    usize next_line = 0;
    // One line of the log arrives: printed, then ingested. Returns the chat
    // when the planner should look.
    const auto feed = [&](const auto &line) {
        sim = std::max(sim, line.t);
        print(std::format("[{}] {}: {}", clock(line.t), line.name, line.text));
        const onebot::MessageEvent event{
            .time = base + sim,
            .self_id = k_replay_self,
            .message_type = "group",
            .message_id = ++message_id,
            .user_id = line.uid,
            .group_id = k_replay_group,
            .group_name = "回放群",
            .sender = {.user_id = line.uid, .nickname = line.name},
            .message = to_segments(line.text, names),
        };
        return responder.ingest(event);
    };
    // A planner wait: the clock moves on, and the lines of that time arrive.
    responder.pause = [&](const store::ChatKey &, i64 seconds) -> Task<> {
        sim += seconds;
        while (next_line < lines.size() && lines[next_line].t <= sim) std::ignore = feed(lines[next_line++]);
        co_return;
    };
    if (continue_at) {
        // What she would say right before line `continue_at`: the log up to
        // there is fed without turns (its last three lines count as new),
        // then one turn runs. Waits do not reveal the lines that follow.
        struct Outcome {
            bool replied = false;
            std::vector<std::string> messages;
            std::vector<std::string> decisions;
        } outcome;
        responder.outbox = [&outcome, &next_id](const store::ChatKey &, onebot::Message message) -> Task<std::optional<i64>> {
            const bool sticker = std::ranges::any_of(message, [](const onebot::Segment &segment) { return segment.type == "image"; });
            outcome.messages.push_back(sticker ? std::string("[表情包]") : onebot::render_plain(message));
            outcome.replied = true;
            co_return ++next_id;
        };
        responder.observer = [&outcome](const store::ChatKey &, const chat::Plan &plan, const std::vector<std::string> &) {
            outcome.decisions.push_back(plan.wait > 0 ? std::format("wait {}: {}", plan.wait, plan.reason) :
                                        plan.reply    ? std::format("reply {}: {}", plan.target, plan.intent) :
                                                        std::format("end: {}", plan.reason));
        };
        responder.pause = [&sim](const store::ChatKey &, i64 seconds) -> Task<> {
            sim += seconds;
            co_return;
        };
        responder.must_speak = true;
        const store::ChatKey chat{.type = "group", .id = k_replay_group};
        const usize stop = std::min(*continue_at, lines.size());
        for (usize i = 0; i < stop; ++i) {
            if (i + 3 == stop || (stop < 3 && i == 0)) {
                auto &state = responder.chats[chat];
                state.flow.unseen.clear();
                state.flow.addressed.reset();
            }
            std::ignore = feed(lines[i]);
        }
        if (!responder.chats[chat].flow.unseen.empty()) co_await responder.run_turn(chat);
        co_await tasks.join();
        // Her messages as stored, so a sticker reads "[表情: 描述]" like anyone else's.
        if (auto recent = store->recent_messages(chat, 40); recent.has_value()) {
            std::vector<std::string> stored;
            for (const auto &message : *recent) {
                if (message.from_self) stored.push_back(message.text);
            }
            if (stored.size() == outcome.messages.size()) outcome.messages = std::move(stored);
        }
        print("RESULT " + glz::write_json(outcome).value_or("{}"));
        co_return 0;
    }
    while (next_line < lines.size()) {
        if (const auto chat = feed(lines[next_line++])) {
            // As turn_worker: look again while messages left from the turn call for it.
            co_await responder.run_turn(*chat);
            auto &state = responder.chats[*chat];
            for (int extra = 0; extra < 3 && chat::should_plan(state.flow, responder.flow_settings(), responder.now()); ++extra)
                co_await responder.run_turn(*chat);
        }
    }
    co_await tasks.join();
    co_return 0;
}

Task<int> run_learn(Config config, std::filesystem::path log) {
    auto parsed = read_log(log);
    if (!parsed) co_return 1;
    auto store = store::Store::open_memory();
    if (store.has_error()) {
        std::fprintf(stderr, "%s\n", store.error().message().c_str());
        co_return 1;
    }
    lighter::TaskGroup tasks(lighter::EventLoop::current());
    learn::Learner learner(config.learning, config.chat.name, config.chat.aliases, *store, make_learner(config.model),
                           [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    const usize batch = std::max<usize>(config.learning.batch_messages, 2);
    std::vector<learn::Line> lines;
    for (const auto &line : *parsed) {
        if (auto text = learn::learnable_text(to_segments(line.text, {})); !text.empty())
            lines.push_back({.sender = line.uid, .text = text});
    }
    for (usize start = 0; start < lines.size(); start += batch) {
        std::vector<learn::Line> chunk(lines.begin() + static_cast<std::ptrdiff_t>(start),
                                       lines.begin() + static_cast<std::ptrdiff_t>(std::min(lines.size(), start + batch)));
        print(std::format("== 第 {}-{} 行", start + 1, start + chunk.size()));
        // A co_await inside the range-for header crashes GCC 16.2.
        const auto learned = co_await learner.learn(1, std::move(chunk));
        for (const auto &pattern : learned) print(std::format("  当{}时，可以{}", pattern.situation, pattern.style));
    }
    co_await tasks.join();
    co_return 0;
}

Task<int> review_expressions(Config config, bool apply, std::vector<std::string> keep) {
    const auto path = std::filesystem::path(config.data_dir) / "lilia.db";
    auto store = apply ? store::Store::open(path) : store::Store::open_readonly(path);
    if (store.has_error()) {
        std::fprintf(stderr, "%s\n", store.error().message().c_str());
        co_return 1;
    }
    auto rows = store->expressions(0);
    if (rows.has_error()) {
        std::fprintf(stderr, "%s\n", rows.error().message().c_str());
        co_return 1;
    }
    // The dry run cannot record its review calls in a read-only database.
    auto scratch = store::Store::open_memory();
    lighter::TaskGroup tasks(lighter::EventLoop::current());
    learn::Learner learner(config.learning, config.chat.name, config.chat.aliases, apply ? *store : *scratch, make_learner(config.model),
                           [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    std::vector<i64> rejected;
    for (usize start = 0; start < rows->size(); start += k_review_batch) {
        const usize end = std::min(rows->size(), start + k_review_batch);
        std::vector<learn::Learned> batch;
        for (usize i = start; i < end; ++i) batch.push_back({.situation = (*rows)[i].situation, .style = (*rows)[i].style});
        // A co_await inside an if/for header crashes GCC 16.2.
        const auto verdicts = co_await learner.review(0, batch);
        if (!verdicts) {
            std::fprintf(stderr, "review call failed; stopping\n");
            co_return 1;
        }
        for (usize i = start; i < end; ++i) {
            const auto &row = (*rows)[i];
            const bool blocked = learn::blocked_expression(row.situation, row.style);
            const bool protected_ =
                std::ranges::any_of(keep, [&](const std::string &text) { return row.style.find(text) != std::string::npos; });
            const auto verdict = std::ranges::find(*verdicts, i - start, &learn::Rejected::index);
            if (protected_ || (!blocked && verdict == verdicts->end())) continue;
            rejected.push_back(row.id);
            print(std::format("  {}当{}时，可以{}{}", blocked ? "[屏蔽] " : "", row.situation, row.style,
                              verdict != verdicts->end() && !verdict->reason.empty() ? "（" + verdict->reason + "）" : ""));
        }
    }
    print(std::format("({} 条中 {} 条不合适)", rows->size(), rejected.size()));
    if (apply) {
        if (auto deleted = store->delete_expressions(rejected); deleted.has_error()) {
            std::fprintf(stderr, "%s\n", deleted.error().message().c_str());
            co_return 1;
        }
        print(std::format("已删除 {} 条", rejected.size()));
    }
    co_await tasks.join();
    co_return 0;
}

int manage_expressions(const std::filesystem::path &database, const Manage &request) {
    auto store = request.action == "rejected" ? store::Store::open_readonly(database) : store::Store::open(database);
    if (store.has_error()) {
        std::fprintf(stderr, "%s\n", store.error().message().c_str());
        return 1;
    }
    const i64 now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    if (request.action == "rejected") {
        auto reviews = store->expression_reviews("rejected", request.limit);
        if (reviews.has_error()) return 1;
        for (const auto &review : *reviews) {
            print(std::format("{:>5}  当{}时，可以{}（{}）", review.id, review.situation, review.style, review.reason));
        }
        print(std::format("({} 条被拒；用 --rescue <编号> 找回)", reviews->size()));
        return 0;
    }
    if (request.action == "rescue") {
        auto review = store->expression_review(request.id);
        if (review.has_error() || !*review) {
            std::fprintf(stderr, "no review %lld\n", static_cast<long long>(request.id));
            return 1;
        }
        const auto &found = **review;
        auto id = store->learn_expression(found.chat, found.situation, found.style, now);
        if (id.has_error()) return 1;
        std::ignore = store->set_expression_checked(*id, true);
        std::ignore = store->add_expression_review({.time = now,
                                                    .chat = found.chat,
                                                    .situation = found.situation,
                                                    .style = found.style,
                                                    .verdict = "rescued",
                                                    .reason = "人工找回"});
        print(std::format("已找回并标记为人工确认：当{}时，可以{}（表达 {}）", found.situation, found.style, *id));
        return 0;
    }
    if (request.action == "check" || request.action == "uncheck") {
        if (auto marked = store->set_expression_checked(request.id, request.action == "check"); marked.has_error()) return 1;
        print(std::format("表达 {} 已{}", request.id, request.action == "check" ? "标记为人工确认" : "取消人工确认"));
        return 0;
    }
    return 2;
}

int manage_jargon(const std::filesystem::path &database, const Manage &request) {
    auto store = request.action == "list" ? store::Store::open_readonly(database) : store::Store::open(database);
    if (store.has_error()) {
        std::fprintf(stderr, "%s\n", store.error().message().c_str());
        return 1;
    }
    const i64 now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    if (request.action == "set") {
        if (auto set = store->set_jargon_manual(request.term, request.meaning, now); set.has_error()) return 1;
        print(std::format("已设定：{}：{}", request.term, request.meaning));
        return 0;
    }
    if (request.action == "delete") {
        if (auto deleted = store->delete_jargon(request.term); deleted.has_error()) return 1;
        print(std::format("已删除：{}", request.term));
        return 0;
    }
    auto terms = store->jargon_terms();
    if (terms.has_error()) return 1;
    usize shown = 0;
    for (const auto &term : *terms) {
        if (shown++ >= request.limit) break;
        const auto verdict = term.is_jargon == 1 ? "黑话" : term.is_jargon == 0 ? "普通词" : "待定";
        print(std::format("{:>4}  {}［{}{}］{}", term.count, term.content, verdict, term.created_by == "MANUAL" ? "，人工" : "",
                          term.meaning.empty() ? "" : "：" + term.meaning));
    }
    print(std::format("({} 个词条)", terms->size()));
    return 0;
}

int print_expressions(const std::filesystem::path &database, usize limit) {
    auto store = store::Store::open_readonly(database);
    if (store.has_error()) {
        std::fprintf(stderr, "%s\n", store.error().message().c_str());
        return 1;
    }
    auto rows = store->expressions(limit);
    if (rows.has_error()) {
        std::fprintf(stderr, "%s\n", rows.error().message().c_str());
        return 1;
    }
    for (const auto &row : *rows) {
        print(std::format("{:>5} ×{:<3} 当{}时，可以{}{}{}", row.id, row.count, row.situation, row.style, row.checked ? "［人工确认］" : "",
                          row.chat.id == 0 ? "" : std::format("［群 {}］", row.chat.id)));
    }
    print(std::format("({} 条)", rows->size()));
    return 0;
}

} // namespace lilia