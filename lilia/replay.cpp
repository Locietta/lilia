#include "replay.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <glaze/json.hpp>

#include <lighter/async/io/watcher.h>
#include <lighter/async/runtime/task_group.h>

#include <lilia/chat/responder.h>
#include <lilia/feedback/feedback.h>
#include <lilia/chat/trigger.h>
#include <lilia/learn/learner.h>
#include <lilia/log.h>
#include <lilia/memory/memory.h>
#include <lilia/retrieval/retriever.h>
#include <lilia/onebot/message.h>
#include <lilia/onebot/session.h>
#include <lilia/provider/anthropic.h>
#include <lilia/store/store.h>

namespace lilia {

using lighter::Task;

namespace {

constexpr i64 k_replay_group = 1;
constexpr i64 k_replay_self = 10'001;
/// Expressions per review call in `lilia expressions --review`.
constexpr usize k_review_batch = 20;

/// What a probe line should get: whom it is said to ("her", "other", "group"
/// or "unsure") and whether she should speak (unset: either is fine).
struct ReplayExpect {
    std::string to_her;
    std::optional<bool> speak;
    std::string note;
};

struct ReplayLine {
    i64 t = 0;
    std::optional<i64> time; ///< when it was said (unix seconds), from export-group-chats.py
    i64 uid = 0;
    std::string name;
    std::string text;
    std::string id;              ///< a label other lines can quote
    std::string reply_to;        ///< the label (or "#n", 1-based) it quotes
    std::vector<std::string> at; ///< names it @s; "她" is her
    bool self = false;           ///< her own earlier line, stored as hers
    std::string scenario;        ///< probes see only their scenario's lines
    std::optional<ReplayExpect> expect;
    /// From export-group-chats.py: the message as QQ sent it (anonymised), so
    /// lilia renders it herself as live. Quotes name a line `id`; an @ of
    /// "self" is of her; pictures have a path under /replay-media, the img/
    /// directory beside the log. `text` is then only shown, not ingested.
    std::optional<onebot::Message> segments;
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

/// A replay line's segments: its quote and @s, then its text (to_segments).
onebot::Message line_segments(const ReplayLine &line, const std::vector<std::string> &names, const std::map<std::string, i64> &ids,
                              const std::map<std::string, i64> &uids) {
    onebot::Message message;
    if (const auto quoted = ids.find(line.reply_to); !line.reply_to.empty() && quoted != ids.end()) {
        message.push_back(onebot::reply_segment(quoted->second));
    }
    for (const auto &name : line.at) {
        if (name == "她") {
            message.push_back(onebot::at_segment(k_replay_self));
            continue;
        }
        const auto uid = uids.find(name);
        auto mention = onebot::at_segment(uid == uids.end() ? 0 : uid->second);
        mention.data["name"] = name;
        message.push_back(std::move(mention));
    }
    for (auto &segment : to_segments(line.text, names)) message.push_back(std::move(segment));
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

/// Claude caches only when asked; `cache` false for prompts that are never
/// sent twice, where a cache write only costs more.
provider::Provider make_client(const ModelConfig &model, const std::string &base_url, const std::string &api_key, const std::string &name,
                               u32 max_tokens, bool require_tool = false, bool bearer = false, const std::string &thinking_off = {},
                               bool cache = true, const std::optional<std::string> &effort = {}) {
    return pro::make_proxy<provider::ProviderFacade, anthropic::Client>(anthropic::ClientOptions{
        .auth = provider::api_key_auth(api_key, bearer),
        .base_url = base_url,
        .model = name,
        .reasoning_effort = effort ? effort : model.reasoning_effort,
        .disable_thinking = !model.thinking,
        .thinking_off = thinking_off.empty() ? "disabled" : thinking_off,
        .require_tool = require_tool,
        .max_tokens = max_tokens,
        .prompt_cache = cache && name.starts_with("claude"),
    });
}

} // namespace

provider::Provider make_planner(const ModelConfig &model) {
    const bool own = !model.planner_base_url.empty();
    return make_client(model, own ? model.planner_base_url : model.base_url, own ? model.planner_api_key : model.api_key,
                       model.planner_model, model.planner_max_tokens, model.planner_force_tool, own && model.planner_bearer,
                       own ? model.planner_thinking_off : std::string(), true, model.planner_reasoning_effort);
}

provider::Provider make_selector(const ModelConfig &model) {
    const bool own = !model.planner_base_url.empty();
    return make_client(model, own ? model.planner_base_url : model.base_url, own ? model.planner_api_key : model.api_key,
                       model.planner_model, model.planner_max_tokens, true, own && model.planner_bearer,
                       own ? model.planner_thinking_off : std::string(), false, model.planner_reasoning_effort);
}

provider::Provider make_warmer(const ModelConfig &model) {
    // The planner's request but one output token: max_tokens is not part of
    // what the cache matches; tool_choice and thinking are, so they stay.
    const bool own = !model.planner_base_url.empty();
    return make_client(model, own ? model.planner_base_url : model.base_url, own ? model.planner_api_key : model.api_key,
                       model.planner_model, 1, model.planner_force_tool, own && model.planner_bearer,
                       own ? model.planner_thinking_off : std::string(), true, model.planner_reasoning_effort);
}

std::vector<chat::Responder::NamedProvider> make_planner_fallbacks(const ModelConfig &model) {
    std::vector<chat::Responder::NamedProvider> fallbacks;
    for (const auto &fallback : model.planner_fallbacks) {
        fallbacks.push_back({
            .model = fallback.model,
            .provider = make_client(model, fallback.base_url.empty() ? model.base_url : fallback.base_url,
                                    fallback.api_key.empty() ? model.api_key : fallback.api_key, fallback.model, model.planner_max_tokens,
                                    model.planner_force_tool, !fallback.api_key.empty() && fallback.bearer, fallback.thinking_off, true,
                                    fallback.reasoning_effort),
        });
    }
    return fallbacks;
}

provider::Provider make_vision(const ModelConfig &model) {
    return make_client(model, model.base_url, model.api_key, model.vision_model, 400);
}

provider::Provider make_learner(const ModelConfig &model) {
    return make_client(model, model.base_url, model.api_key, model.helper(), 1'000, true);
}

provider::Provider make_judge(const ModelConfig &model) {
    // A verdict for each of up to 60 lines does not fit the learner's 1000 tokens.
    return make_client(model, model.base_url, model.api_key, model.helper(), 6'000, true);
}

provider::Provider make_summarizer(const ModelConfig &model) {
    // A window of 40 to 80 lines with pictures and her stances ran past the
    // learner's 1000 tokens, and the cut tool call failed to parse.
    return make_client(model, model.base_url, model.api_key, model.helper(), 2'000, true);
}

provider::Provider make_searcher(const ModelConfig &model) {
    return make_client(model, model.base_url, model.api_key, model.helper(), 1'200);
}

provider::Provider make_writer(const ModelConfig &model) {
    return pro::make_proxy<provider::ProviderFacade, anthropic::Client>(anthropic::ClientOptions{
        .auth = provider::api_key_auth(model.writer_api_key.empty() ? model.api_key : model.writer_api_key,
                                       !model.writer_api_key.empty() && model.writer_bearer),
        .base_url = model.writer_base_url.empty() ? model.base_url : model.writer_base_url,
        .model = model.model,
        .reasoning_effort = model.writer_reasoning_effort ? model.writer_reasoning_effort : model.reasoning_effort,
        .disable_thinking = !model.thinking,
        .thinking_off = model.writer_thinking_off.empty() ? "disabled" : model.writer_thinking_off,
        .temperature = model.writer_temperature,
        .max_tokens = model.max_tokens,
        .prompt_cache = model.model.starts_with("claude"),
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
                .reasoning_effort = fallback.reasoning_effort ? fallback.reasoning_effort : model.reasoning_effort,
                .disable_thinking = !model.thinking,
                .thinking_off = fallback.thinking_off.empty() ? "disabled" : fallback.thinking_off,
                .temperature = model.writer_temperature,
                .max_tokens = model.max_tokens,
                .prompt_cache = fallback.model.starts_with("claude"),
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

/// A prompt variant's text; nullopt (reported) when it cannot be read.
std::optional<std::string> read_prompt(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot read %s\n", path.string().c_str());
        return std::nullopt;
    }
    return std::string(std::istreambuf_iterator<char>(input), {});
}

/// A responder for one probe run: the persona, fallbacks and planner prompt
/// the run asks for, without delays. Null (reported) when the persona cannot
/// be read.
std::unique_ptr<chat::Responder> probe_responder(const Config &config, store::Store &store, chat::Responder::Spawn spawn,
                                                 const std::optional<std::filesystem::path> &persona, const std::string &planner_prompt) {
    auto responder =
        std::make_unique<chat::Responder>(config, store, make_planner(config.model), make_searcher(config.model), make_writer(config.model),
                                          persona.value_or(std::filesystem::path(config.data_dir) / "persona.md"), std::move(spawn));
    if (auto loaded = responder->reload_persona(); loaded.has_error()) {
        std::fprintf(stderr, "%s\n", loaded.error().message().c_str());
        return nullptr;
    }
    responder->writer_fallbacks = make_writer_fallbacks(config.model);
    responder->planner_fallbacks = make_planner_fallbacks(config.model);
    responder->selector = make_selector(config.model);
    responder->planner_prompt = planner_prompt;
    responder->realtime = false;
    return responder;
}

} // namespace

Task<int> run_replay(Config config, std::filesystem::path log, std::optional<std::filesystem::path> persona,
                     std::optional<std::filesystem::path> habits, std::optional<std::filesystem::path> writer_prompt,
                     std::optional<std::filesystem::path> stickers, std::optional<usize> continue_at, bool forced) {
    auto parsed = read_log(log);
    if (!parsed) co_return 1;
    auto lines = *std::move(parsed);

    config.chat.groups = {k_replay_group};
    config.chat.private_users.clear();
    config.chat.admins.clear();
    config.feedback.enabled = false; // the prior is shown, nothing is judged
    // Logged pictures come from the files beside the log, through the archiver as live.
    config.archive.enabled = true;
    config.archive.container_qq_dir = "/replay-media";
    config.archive.host_qq_dir = (log.parent_path() / "img").string();
    // The replay chat stands in for the groups she lives in: it shares what
    // they learned (a live config lists only real groups).
    for (auto *groups : {&config.learning.expression_groups, &config.learning.jargon_groups}) {
        for (auto &group : *groups) group.push_back(k_replay_group);
    }
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
        std::vector<std::string> hashes;
        for (const auto &row : *rows) {
            for (i64 i = 0; i < row.count; ++i) std::ignore = store->learn_expression(row.chat, row.situation, row.style, row.last_active);
            hashes.push_back(retrieval::text_hash(learn::expression_text(row.situation, row.style)));
        }
        // Their vectors too, so a replay does not embed them again.
        const auto profile = retrieval::profile_of(config.memory);
        if (auto vectors = source->embeddings(profile, hashes); vectors.has_value()) std::ignore = store->add_embeddings(profile, *vectors);
        print(std::format("(载入 {} 条学来的说法)", rows->size()));
        // The slang she knows, in the chats it is known in, and the fitted
        // addressee prior, as live.
        usize slang = 0;
        if (auto terms = source->jargon_terms(); terms.has_value()) {
            for (const auto &term : *terms) {
                if (term.is_jargon != 1 || term.meaning.empty()) continue;
                auto chats = source->jargon_chats(term.id);
                if (!chats.has_value() || chats->empty()) continue;
                i64 id = 0;
                for (const auto chat_id : *chats) {
                    if (auto copy = store->add_jargon_hit(term.content, {.type = "group", .id = chat_id}, "", term.last_seen, 0);
                        copy.has_value()) {
                        id = copy->id;
                    }
                }
                if (id != 0 && store->set_jargon_inference(id, term.meaning, term.is_jargon, term.count, true).has_value()) ++slang;
            }
        }
        if (auto models = source->addressee_models(); models.has_value()) {
            for (const auto &model : *models) std::ignore = store->save_addressee_model(model);
        }
        print(std::format("(载入 {} 条黑话)", slang));
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
        auto rules = read_prompt(*writer_prompt);
        if (!rules) co_return 1;
        responder.writer_rules = *std::move(rules);
    }
    responder.writer_fallbacks = make_writer_fallbacks(config.model);
    responder.planner_fallbacks = make_planner_fallbacks(config.model);
    responder.selector = make_selector(config.model);
    // The addressee prior as live: the fitted models copied above, else its hand-set weights.
    feedback::Feedback feedback(config.feedback, config.chat, *store, make_judge(config.model),
                                [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    feedback.load();
    responder.feedback = &feedback;
    sticker::Collector collector(config.stickers, config.data_dir, *store, [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    if (stickers) {
        auto source = store::Store::open_readonly(*stickers);
        auto pool = source.has_value() ? source->stickers("registered") :
                                         Result<std::vector<store::Sticker>>(lighter::outcome_error(source.error()));
        if (pool.has_error()) {
            std::fprintf(stderr, "%s\n", pool.error().message().c_str());
            co_return 1;
        }
        // The rest of the collection too: a sticker someone posts reads as its
        // learned description whatever its status, as live.
        for (const auto status : {"spare", "pending", "rejected"}) {
            if (auto others = source->stickers(status); others.has_value()) {
                for (auto &sticker : *others) std::ignore = store->add_sticker(sticker);
            }
        }
        std::vector<std::string> hashes;
        for (auto sticker : *pool) {
            sticker.path = (stickers->parent_path() / sticker.path).string(); // absolute: read from the live data dir
            std::ignore = store->add_sticker(sticker);
            hashes.push_back(retrieval::text_hash(sticker::embedding_text(sticker)));
            if (!sticker.usage.empty()) hashes.push_back(retrieval::text_hash(sticker::usage_text(sticker)));
        }
        // Their vectors too, so a replay does not embed the pool again.
        const auto profile = retrieval::profile_of(config.memory);
        if (auto vectors = source->embeddings(profile, hashes); vectors.has_value()) std::ignore = store->add_embeddings(profile, *vectors);
        responder.stickers = &collector;
        print(std::format("(载入 {} 个表情包)", pool->size()));
    }
    responder.self = {.user_id = k_replay_self, .nickname = config.chat.name};
    responder.realtime = false;
    archive::Archiver archiver(config.archive, config.data_dir, *store);
    responder.archiver = &archiver;

    // The chat plays out today at the time of day it was said (every line
    // shows its time), whenever the replay runs; a log with relative times
    // only starts at 20:00 (a run at 3 am had her remark on the hour in chats
    // that were never that late).
    const i64 offset = i64{config.chat.utc_offset_minutes} * 60;
    const i64 wall = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const i64 start = lines.empty() || !lines.front().time ? 20 * 3600 : ((*lines.front().time + offset) % 86'400 + 86'400) % 86'400;
    const i64 base = (wall + offset) / 86'400 * 86'400 - offset + start - (lines.empty() ? 0 : lines.front().t);
    i64 sim = 0;
    responder.now = [&sim, base] { return base + sim; };
    retrieval::Retriever retriever(config.memory, *store);
    memory::Memory memory(config.memory, config.chat.name, config.chat.utc_offset_minutes, *store, retriever);
    if (config.memory.enabled) {
        memory.now = [&sim, base] { return base + sim; };
        memory.summarizer = make_summarizer(config.model);
        memory.summarizer_model = config.model.helper();
        memory.data_dir = config.data_dir;
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
        if (!plan.analysis.empty()) print(std::format("        ~ {}", plan.analysis));
        if (plan.wait > 0) {
            print(std::format("        · 等 {} 秒：{}", plan.wait, plan.reason));
        } else if (plan.repeat || plan.intent == "复读") {
            print("        · 复读");
        } else if (!plan.sticker.empty()) {
            print(std::format("        · 表情包 {}", plan.sticker));
        } else if (plan.poke) {
            print("        · 戳一戳");
        } else if (!plan.reply) {
            print(std::format("        · {}：{}", plan.follow_up ? "到此为止" : "不说话", plan.reason));
        } else {
            print(
                std::format("        · 回复 {}（{}{}）：{}", plan.target, style_name(plan.style), plan.quote ? "，引用" : "", plan.intent));
        }
    };

    std::vector<std::string> names = config.chat.aliases;
    names.push_back(config.chat.name);
    i64 message_id = 0;
    usize next_line = 0;
    std::map<std::string, i64> ids; // quotable labels: "#n" and each line's id
    usize fed = 0;
    std::map<std::string, i64> uids;
    for (const auto &line : lines) {
        if (!line.self) uids.emplace(line.name, line.uid);
    }
    // A logged message as it came, its quote and @s pointed at this replay's ids.
    const auto remap = [&ids](onebot::Message segments) {
        for (auto &segment : segments) {
            if (segment.type == "reply") {
                const auto found = ids.find(onebot::data_string(segment, "id").value_or(""));
                segment = onebot::reply_segment(found == ids.end() ? -1 : found->second);
            } else if (segment.type == "at" && onebot::data_string(segment, "qq") == "self") {
                segment = onebot::at_segment(k_replay_self);
            }
        }
        return segments;
    };
    const store::ChatKey replay_chat{.type = "group", .id = k_replay_group};
    // One line of the log arrives: printed, then ingested (hers are stored as
    // said before). Returns the chat when the planner should look.
    const auto feed = [&](const ReplayLine &line) -> std::optional<store::ChatKey> {
        sim = std::max(sim, line.t);
        const i64 id = ++message_id;
        auto message = line.segments ? remap(*line.segments) : line_segments(line, names, ids, uids);
        ids["#" + std::to_string(++fed)] = id;
        if (!line.id.empty()) ids[line.id] = id;
        if (line.self) {
            print(std::format("[{}]   {}: {}", clock(line.t), config.chat.name, line.text));
            // Her quote rendered as anyone's would be.
            std::string quote;
            if (const auto target = onebot::reply_target(message)) {
                auto quoted = store->message(replay_chat, *target);
                quote = chat::quote_prefix(quoted.has_value() && *quoted ? &**quoted : nullptr, config.chat.name);
            }
            const store::StoredMessage stored{.chat = replay_chat,
                                              .message_id = id,
                                              .sender_id = k_replay_self,
                                              .sender_name = config.chat.name,
                                              .time = base + sim,
                                              .segments_json = glz::write_json(message).value_or("[]"),
                                              .text = quote + onebot::render_plain(message),
                                              .from_self = true};
            std::ignore = store->add_message(stored);
            if (archive::needs_enrichment(message, true)) {
                archiver.track(replay_chat, id);
                tasks.spawn(archiver.archive(nullptr, stored, message, quote));
            }
            return std::nullopt;
        }
        print(std::format("[{}] {}: {}", clock(line.t), line.name, line.text));
        const onebot::MessageEvent event{
            .time = base + sim,
            .self_id = k_replay_self,
            .message_type = "group",
            .message_id = id,
            .user_id = line.uid,
            .group_id = k_replay_group,
            .group_name = "回放群",
            .sender = {.user_id = line.uid, .nickname = line.name},
            .message = std::move(message),
        };
        return responder.ingest(event);
    }; // A planner wait: the clock moves on, and the lines of that time arrive.
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
        // Forced (blind comparisons) she must say something; free, she may stay silent as live.
        responder.must_speak = forced;
        const store::ChatKey chat{.type = "group", .id = k_replay_group};
        const usize stop = std::min(*continue_at, lines.size());
        for (usize i = 0; i < stop; ++i) {
            if (i + 3 == stop || (stop < 3 && i == 0)) {
                auto &state = responder.chats[chat];
                state.flow.unseen.clear();
                state.flow.addressed.reset();
                state.flow.named.reset();
            }
            std::ignore = feed(lines[i]);
        }
        // As live by the time she looks: every picture archived, the chat
        // indexed for search, what lies above her window summarised.
        std::vector<i64> fed_ids;
        for (i64 id = 1; id <= message_id; ++id) fed_ids.push_back(id);
        co_await archiver.wait_for(chat, std::move(fed_ids), std::chrono::seconds(60));
        if (responder.memory) {
            // index() only asks again while the one ingest started still runs;
            // live, the history is indexed and summarised long before.
            co_await memory.index(chat);
            while (std::ranges::contains(memory.indexing, chat)) co_await lighter::sleep(std::chrono::milliseconds(100));
            responder.memory = nullptr; // the window without the summary log_window would spawn
            const auto window = responder.log_window(chat);
            responder.memory = &memory;
            if (!window.empty()) co_await memory.summarize_window(chat, window.front().row_id - 1, 2 * config.chat.context_messages);
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
        // No timers here: lines held back by the idle backoff wait for the next line.
        if (const auto chat = feed(lines[next_line++]);
            chat && chat::should_plan(responder.chats[*chat].flow, responder.flow_settings(), responder.now())) {
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

Task<int> run_probes(Config config, std::filesystem::path log, std::optional<std::filesystem::path> persona, usize repeat,
                     std::optional<std::filesystem::path> planner_prompt) {
    auto parsed = read_log(log);
    if (!parsed) co_return 1;
    std::string planner_text;
    if (planner_prompt) {
        auto text = read_prompt(*planner_prompt);
        if (!text) co_return 1;
        planner_text = *std::move(text);
    }
    const auto lines = *std::move(parsed);
    config.chat.groups = {k_replay_group};
    config.chat.private_users.clear();
    config.chat.admins.clear();
    config.feedback.enabled = false; // the prior is shown, nothing is judged
    // Logged pictures come from the files beside the log, through the archiver as live.
    config.archive.enabled = true;
    config.archive.container_qq_dir = "/replay-media";
    config.archive.host_qq_dir = (log.parent_path() / "img").string();
    std::vector<std::string> names = config.chat.aliases;
    names.push_back(config.chat.name);
    std::map<std::string, i64> uids;
    for (const auto &line : lines) {
        if (!line.self) uids.emplace(line.name, line.uid);
    }
    const store::ChatKey chat{.type = "group", .id = k_replay_group};
    usize intrusions = 0;
    usize should_be_silent = 0;
    usize misses = 0;
    usize should_speak = 0;
    usize agreed = 0;
    usize judged = 0;
    for (usize probe = 0; probe < lines.size(); ++probe) {
        if (!lines[probe].expect) continue;
        const auto &expect = *lines[probe].expect;
        std::vector<usize> scenario;
        for (usize i = 0; i <= probe; ++i) {
            if (lines[i].scenario == lines[probe].scenario) scenario.push_back(i);
        }
        for (usize run = 0; run < std::max<usize>(repeat, 1); ++run) {
            auto store = store::Store::open_memory();
            if (store.has_error()) co_return 1;
            lighter::TaskGroup tasks(lighter::EventLoop::current());
            const auto made =
                probe_responder(config, *store, [&tasks](Task<> task) { tasks.spawn(std::move(task)); }, persona, planner_text);
            if (!made) co_return 1;
            auto &responder = *made;
            responder.self = {.user_id = k_replay_self, .nickname = config.chat.name};
            // The addressee prior as live, from its hand-set weights.
            feedback::Feedback feedback(config.feedback, config.chat, *store, make_judge(config.model),
                                        [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
            responder.feedback = &feedback;
            const i64 base = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            i64 sim = 0;
            responder.now = [&sim, base] { return base + sim; };
            responder.pause = [&sim](const store::ChatKey &, i64 seconds) -> Task<> {
                sim += seconds; // a wait reveals nothing: the probe ends at its line
                co_return;
            };
            bool spoke = false;
            std::string to_her;
            std::string said;
            i64 next_id = 1'000'000;
            responder.outbox = [&](const store::ChatKey &, onebot::Message message) -> Task<std::optional<i64>> {
                spoke = true;
                said += (said.empty() ? "" : " / ") + onebot::render_plain(message);
                co_return ++next_id;
            };
            responder.observer = [&](const store::ChatKey &, const chat::Plan &plan, const std::vector<std::string> &) {
                if (plan.reply && to_her.empty()) to_her = plan.to_her;
            };
            std::map<std::string, i64> ids;
            i64 message_id = 0;
            usize fed = 0;
            for (usize k = 0; k < scenario.size(); ++k) {
                const auto &line = lines[scenario[k]];
                // Only the last three lines are new to the turn.
                if (k + 3 == scenario.size() || (scenario.size() < 3 && k == 0)) {
                    auto &state = responder.chats[chat];
                    state.flow.unseen.clear();
                    state.flow.addressed.reset();
                    state.flow.named.reset();
                }
                sim = std::max(sim, line.t);
                const i64 id = ++message_id;
                auto message = line_segments(line, names, ids, uids);
                ids["#" + std::to_string(++fed)] = id;
                if (!line.id.empty()) ids[line.id] = id;
                if (line.self) {
                    std::ignore = store->add_message({.chat = chat,
                                                      .message_id = id,
                                                      .sender_id = k_replay_self,
                                                      .sender_name = config.chat.name,
                                                      .time = base + sim,
                                                      .segments_json = glz::write_json(message).value_or("[]"),
                                                      .text = onebot::render_plain(message),
                                                      .from_self = true});
                    continue;
                }
                std::ignore = responder.ingest({.time = base + sim,
                                                .self_id = k_replay_self,
                                                .message_type = "group",
                                                .message_id = id,
                                                .user_id = line.uid,
                                                .group_id = k_replay_group,
                                                .group_name = "回放群",
                                                .sender = {.user_id = line.uid, .nickname = line.name},
                                                .message = std::move(message)});
            }
            if (!responder.chats[chat].flow.unseen.empty()) co_await responder.run_turn(chat);
            co_await tasks.join();

            const bool her = to_her == "called" || to_her == "likely";
            std::string verdict;
            if (expect.speak == false) {
                ++should_be_silent;
                if (spoke) {
                    ++intrusions;
                    verdict += " INTRUSION";
                }
            } else if (expect.speak == true) {
                ++should_speak;
                if (!spoke) {
                    ++misses;
                    verdict += " MISS";
                }
            }
            if (spoke && !to_her.empty() && to_her != "unsure" && expect.to_her != "unsure" && !expect.to_her.empty()) {
                ++judged;
                if (her == (expect.to_her == "her")) {
                    ++agreed;
                } else {
                    verdict += " ADDRESSEE";
                }
            }
            print(std::format("PROBE {} line {}: expect {}/{}, got {}/{}{}{}", lines[probe].scenario, probe + 1, expect.to_her,
                              expect.speak ? (*expect.speak ? "speak" : "silent") : "either", spoke ? "spoke" : "silent",
                              to_her.empty() ? "-" : to_her, verdict, spoke ? "  「" + said + "」" : std::string()));
        }
    }
    print(std::format("SUMMARY intrusions {}/{}, misses {}/{}, to_her agreement {}/{}", intrusions, should_be_silent, misses, should_speak,
                      agreed, judged));
    co_return 0;
}

std::vector<store::AddresseeLabel> pick_probe_labels(std::vector<store::AddresseeLabel> labels, usize limit) {
    std::erase_if(labels, [](const store::AddresseeLabel &label) { return label.chat.type != "group"; });
    if (limit == 0) return labels;
    const auto to_her = static_cast<usize>(std::ranges::count_if(labels, &store::AddresseeLabel::label));
    usize others = limit > to_her ? limit - to_her : 0;
    std::vector<bool> keep(labels.size());
    for (usize i = labels.size(); i-- > 0;) {
        if (labels[i].label) {
            keep[i] = true;
        } else if (others > 0) {
            keep[i] = true;
            --others;
        }
    }
    std::vector<store::AddresseeLabel> picked;
    for (usize i = 0; i < labels.size(); ++i) {
        if (keep[i]) picked.push_back(std::move(labels[i]));
    }
    return picked;
}

chat::Flow probe_flow(const std::vector<store::StoredMessage> &rows, i64 self_id, const ChatConfig &config, store::Store &source) {
    chat::Flow flow;
    usize first = rows.size();
    for (usize others = 0; first > 0 && others < 3;) {
        const auto &row = rows[--first];
        others += !row.from_self && !chat::is_notice(row);
    }
    for (usize i = first; i < rows.size(); ++i) {
        const auto &row = rows[i];
        if (row.from_self || chat::is_notice(row)) continue;
        onebot::Message segments;
        std::ignore = glz::read<onebot::k_read_opts>(segments, row.segments_json);
        // The event ingest classified, as far as the stored row tells.
        bool replies_to_self = false;
        if (const auto target = onebot::reply_target(segments)) {
            auto quoted = source.message(row.chat, *target);
            replies_to_self = quoted.has_value() && *quoted && (*quoted)->from_self;
        }
        const onebot::MessageEvent event{
            .time = row.time,
            .self_id = self_id,
            .message_type = row.chat.type,
            .message_id = row.message_id,
            .user_id = row.sender_id,
            .group_id = row.chat.id,
            .sender = {.user_id = row.sender_id, .nickname = row.sender_name},
            .message = std::move(segments),
        };
        const auto reason = chat::classify(event, self_id, replies_to_self, config);
        if (reason != chat::TriggerReason::NONE) (reason == chat::TriggerReason::ALIAS ? flow.named : flow.addressed) = row.message_id;
        flow.unseen.push_back(row);
    }
    return flow;
}

i64 self_id_of(store::Store &store, const std::vector<i64> &groups) {
    for (const auto group : groups) {
        auto recent = store.recent_messages({.type = "group", .id = group}, 500);
        if (!recent.has_value()) continue;
        for (const auto &message : *recent) {
            if (message.from_self) return message.sender_id;
        }
    }
    return 0;
}

Task<int> run_db_probes(Config config, std::filesystem::path database, std::optional<std::filesystem::path> persona, usize repeat,
                        usize limit, std::optional<std::filesystem::path> planner_prompt) {
    // Private conversations: the info log would show the planner's analyses
    // and what she said.
    if (static_cast<int>(log::g_threshold) > static_cast<int>(log::Level::WARN)) log::g_threshold = log::Level::WARN;
    config.feedback.enabled = false;
    std::string planner_text;
    if (planner_prompt) {
        auto text = read_prompt(*planner_prompt);
        if (!text) co_return 1;
        planner_text = *std::move(text);
    }
    auto source = store::Store::open_readonly(database);
    auto labels = source.has_value() ? source->addressee_labels(0) :
                                       Result<std::vector<store::AddresseeLabel>>(lighter::outcome_error(source.error()));
    if (labels.has_error()) {
        std::fprintf(stderr, "%s\n", labels.error().message().c_str());
        co_return 1;
    }
    const auto picked = pick_probe_labels(*std::move(labels), limit);
    std::vector<i64> groups;
    for (const auto &label : picked) {
        if (!std::ranges::contains(groups, label.chat.id)) groups.push_back(label.chat.id);
    }
    const i64 self_id = self_id_of(*source, groups);
    if (self_id == 0) std::fprintf(stderr, "her QQ id is unknown (she never spoke there): @s of her are not recognised\n");
    config.chat.private_users.clear();
    config.chat.admins.clear();
    const usize window = config.chat.context_messages > 0 ? config.chat.context_messages : 40;
    usize agreed = 0;
    usize judged = 0;
    usize misread = 0;
    std::map<std::string, usize> misread_by; ///< whom the misread lines were said to ("?": not recorded)
    usize group_lines = 0;                   ///< answered lines said to everyone
    usize group_read = 0;                    ///< of them, read as "group"
    usize missed_her = 0;
    usize misses = 0;
    usize to_her_runs = 0;
    usize chip_ins = 0;
    usize other_runs = 0;
    for (const auto &label : picked) {
        const auto &chat = label.chat;
        // "label 0/group": whom it was said to, when the label records it.
        const auto tag = std::format("PROBE chat#{:03} row {} label {}{}", (chat.id < 0 ? -chat.id : chat.id) % 1000, label.row_id,
                                     label.label ? 1 : 0, label.addressee.empty() ? "" : "/" + label.addressee);
        auto line = source->message(chat, label.message_id);
        if (!line.has_value() || !*line) {
            print(tag + ": not stored");
            continue;
        }
        auto rows = source->messages_before(chat, (**line).row_id, window);
        if (rows.has_error()) {
            std::fprintf(stderr, "%s\n", rows.error().message().c_str());
            co_return 1;
        }
        rows->push_back(**line);
        config.chat.groups = {chat.id};
        for (usize run = 0; run < std::max<usize>(repeat, 1); ++run) {
            auto store = store::Store::open_memory();
            if (store.has_error()) co_return 1;
            lighter::TaskGroup tasks(lighter::EventLoop::current());
            const auto made =
                probe_responder(config, *store, [&tasks](Task<> task) { tasks.spawn(std::move(task)); }, persona, planner_text);
            if (!made) co_return 1;
            auto &responder = *made;
            responder.self = {.user_id = self_id, .nickname = config.chat.name};
            // The prior as live (feedback.hint), but from its hand-set weights:
            // nothing is loaded, as models fitted on these very labels would
            // grade themselves.
            feedback::Feedback feedback(config.feedback, config.chat, *store, make_judge(config.model),
                                        [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
            responder.feedback = &feedback;
            i64 sim = (**line).time;
            responder.now = [&sim] { return sim; };
            responder.pause = [&sim](const store::ChatKey &, i64 seconds) -> Task<> {
                sim += seconds; // a wait reveals nothing: the probe ends at its line
                co_return;
            };
            bool spoke = false;
            std::optional<chat::Plan> reply;
            i64 next_id = 1'000'000;
            responder.outbox = [&](const store::ChatKey &, onebot::Message) -> Task<std::optional<i64>> {
                spoke = true;
                co_return ++next_id;
            };
            responder.observer = [&](const store::ChatKey &, const chat::Plan &plan, const std::vector<std::string> &) {
                if (plan.reply && !reply) reply = plan;
            };
            for (const auto &row : *rows) std::ignore = store->add_message(row);
            auto &state = responder.chats[chat];
            state.name = "回放群";
            state.flow = probe_flow(*rows, self_id, config.chat, *source);
            if (!state.flow.unseen.empty()) co_await responder.run_turn(chat);
            co_await tasks.join();

            const bool on_line = reply && reply->target == label.message_id;
            const auto to_her = reply && !reply->to_her.empty() ? reply->to_her : std::string("-");
            std::string verdict;
            if (on_line && (to_her == "called" || to_her == "likely" || to_her == "group" || to_her == "no")) {
                ++judged;
                const bool her = to_her == "called" || to_her == "likely";
                if (label.addressee == "group") {
                    ++group_lines;
                    group_read += to_her == "group";
                }
                if (her == label.label) {
                    ++agreed;
                } else if (her) {
                    ++misread;
                    ++misread_by[label.addressee.empty() ? "?" : label.addressee];
                    verdict += " MISREAD";
                } else {
                    ++missed_her;
                    verdict += " MISSED_HER";
                }
            }
            if (label.label) {
                ++to_her_runs;
                if (!spoke) {
                    ++misses;
                    verdict += " MISS";
                }
            } else {
                ++other_runs;
                chip_ins += spoke;
            }
            print(std::format("{}: {} target {} to_her {}{}", tag, spoke ? "spoke" : "silent",
                              !reply  ? "-" :
                              on_line ? "self" :
                                        "other",
                              to_her, verdict));
        }
    }
    std::string kinds;
    for (const auto &[addressee, count] : misread_by) kinds += std::format("{}{} {}", kinds.empty() ? " (" : ", ", addressee, count);
    if (!kinds.empty()) kinds += ")";
    const auto group = group_lines > 0 ? std::format(", lines to everyone read so {}/{}", group_read, group_lines) : std::string();
    print(std::format(
        "SUMMARY real: to_her agreement {}/{} (misread {}{}, missed_her {}), misses {}/{} on lines to her, chip-in {}/{} on other "
        "lines{}",
        agreed, judged, misread, kinds, missed_her, misses, to_her_runs, chip_ins, other_runs, group));
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

namespace {

struct GroupEntry {
    i64 group_id = 0;
    std::string group_name;
};

std::string clock_text(i64 time, i32 utc_offset_minutes) {
    const auto local = std::chrono::sys_seconds(std::chrono::seconds(time + i64{utc_offset_minutes} * 60));
    return std::format("{:%m-%d %H:%M}", local);
}

Task<int> learn_history(Config config, std::vector<i64> groups, usize count, std::shared_ptr<onebot::Session> session) {
    const auto close = [&session] { session->close(); };
    auto login = co_await session->get_login_info();
    if (login.has_error()) {
        std::fprintf(stderr, "get_login_info failed: %s\n", login.error().message().c_str());
        close();
        co_return 1;
    }
    std::map<i64, std::string> group_names;
    auto listed = co_await session->call("get_group_list", "{}");
    if (listed.has_value()) {
        std::vector<GroupEntry> entries;
        if (!glz::read<onebot::k_read_opts>(entries, *listed)) {
            for (const auto &entry : entries) group_names[entry.group_id] = entry.group_name;
        }
    }
    if (groups.empty()) {
        for (const auto &[id, name] : group_names) groups.push_back(id);
    }

    const auto path = std::filesystem::path(config.data_dir) / "lilia.db";
    auto store = store::Store::open(path);
    if (store.has_error()) {
        std::fprintf(stderr, "%s\n", store.error().message().c_str());
        close();
        co_return 1;
    }
    lighter::TaskGroup tasks(lighter::EventLoop::current());
    learn::Learner learner(config.learning, config.chat.name, config.chat.aliases, *store, make_learner(config.model),
                           [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    learner.frequency_chats = config.chat.groups;
    retrieval::Retriever retriever(config.memory, *store);
    if (config.memory.enabled) learner.retriever = &retriever;
    const usize batch = std::max<usize>(config.learning.batch_messages, 2);
    const i32 offset = config.chat.utc_offset_minutes;

    for (const auto group : groups) {
        const auto name = group_names.contains(group) ? group_names[group] : std::string("?");
        if (std::ranges::contains(config.learning.exclude_groups, group)) {
            print(std::format("== 群 {}（{}）：不学习，跳过", group, name));
            continue;
        }
        const auto messages = co_await onebot::group_history(session, group, count);
        auto span = store->learned_span(group);
        const auto known = span.has_value() ? *span : std::optional<store::LearnedSpan>{};
        std::vector<learn::Line> lines;
        usize covered = 0;
        for (const auto &message : messages) {
            if (known && message.time >= known->first_time && message.time <= known->last_time) {
                ++covered;
                continue;
            }
            auto text = learn::learnable_text(message.message);
            if (text.empty()) continue;
            lines.push_back(
                {.sender = message.user_id, .text = std::move(text), .self = message.user_id == login->user_id, .time = message.time});
        }
        print(std::format("== 群 {}（{}）：取到 {} 条{}，已学过 {} 条，新学 {} 行", group, name, messages.size(),
                          messages.empty() ? std::string() :
                                             std::format("（{} 至 {}）", clock_text(messages.front().time, offset),
                                                         clock_text(messages.back().time, offset)),
                          covered, lines.size()));
        // Older and newer than the learned span are learned as separate runs, never one batch across it.
        usize start = 0;
        while (start < lines.size()) {
            usize end = std::min(lines.size(), start + batch);
            if (known) {
                for (usize i = start + 1; i < end; ++i) {
                    if ((lines[i - 1].time < known->first_time) != (lines[i].time < known->first_time)) {
                        end = i;
                        break;
                    }
                }
            }
            std::vector<learn::Line> chunk(lines.begin() + static_cast<std::ptrdiff_t>(start),
                                           lines.begin() + static_cast<std::ptrdiff_t>(end));
            start = end;
            if (std::ranges::count_if(chunk, [](const learn::Line &line) { return !line.self; }) < 2) continue;
            // A co_await inside the range-for header crashes GCC 16.2.
            const auto learned = co_await learner.learn(group, std::move(chunk));
            for (const auto &pattern : learned) print(std::format("  当{}时，可以{}", pattern.situation, pattern.style));
        }
    }
    // Slang inference runs in the background; let it finish before leaving.
    co_await tasks.join();
    close();
    co_return 0;
}

Task<> drain(std::shared_ptr<onebot::Session> session) {
    // Events of the live groups arrive here too; this run ignores them.
    std::ignore = co_await session->run([](onebot::Event) {});
}

} // namespace

Task<int> run_learn_history(Config config, std::vector<i64> groups, usize count) {
    if (!config.learning.enabled) {
        std::fprintf(stderr, "learning is disabled in the config\n");
        co_return 1;
    }
    auto opened = co_await onebot::Session::open({
        .connect = {.host = config.onebot.host, .port = config.onebot.port, .path = config.onebot.path},
        .access_token = config.onebot.access_token,
        .action_timeout = std::chrono::seconds(60),
    });
    if (opened.has_error()) {
        std::fprintf(stderr, "cannot connect to OneBot at %s:%d: %s\n", config.onebot.host.c_str(), static_cast<int>(config.onebot.port),
                     opened.error().message().c_str());
        co_return 1;
    }
    auto session = *std::move(opened);
    lighter::TaskGroup connection(lighter::EventLoop::current());
    connection.spawn(drain(session));
    const int status = co_await learn_history(std::move(config), std::move(groups), count, session);
    co_await connection.join();
    co_return status;
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