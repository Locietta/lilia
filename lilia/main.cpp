#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <glaze/json.hpp>
#include <xmake/version/lilia.hpp>

#include <lighter/async/async.h>
#include <lighter/async/io/loop.h>
#include <lighter/async/runtime/interrupt.h>

#include <lilia/archive/archiver.h>
#include <lilia/chat/delivery.h>
#include <lilia/chat/responder.h>
#include <lilia/chat/typos.h>
#include <lilia/config.h>
#include <lilia/learn/frequency.h>
#include <lilia/learn/learner.h>
#include <lilia/learn/segment.h>
#include <lilia/log.h>
#include <lilia/onebot/session.h>
#include <lilia/replay.h>
#include <lilia/retrieval/retriever.h>
#include <lilia/sticker/sticker.h>
#include <lilia/store/store.h>

namespace {

using namespace lilia;
using lighter::Task;
using namespace std::chrono_literals;

void print_usage(FILE *output) {
    std::fputs("usage: lilia [--config <path>] [--check-config]\n"
               "       lilia replay <chat.jsonl> [--config <path>] [--persona <persona.md>] [--habits <lilia.db>]\n"
               "                    [--writer-prompt <writer.md>] [--stickers <lilia.db>] [--continue-at <line>]\n"
               "       lilia learn <chat.jsonl> [--config <path>]\n"
               "       lilia learn-history [--group <id>]... [--count <n>] [--config <path>]\n"
               "       lilia expressions [--config <path>] [--limit <n>]\n"
               "       lilia expressions --review [--apply] [--keep <text>]... [--config <path>]\n"
               "       lilia expressions --rejected | --rescue <review id> | --check <id> | --uncheck <id>\n"
               "       lilia jargon [--limit <n>] | --set <term> <meaning> | --delete <term>\n"
               "       lilia retrieval [--query <text>]... [--labels <labels.json>] [--db <lilia.db>] [--config <path>]\n"
               "       lilia stickers --db <copy.db> [--retag [--gifs] [--[no-]frames] [--prompt <sticker.md>] [--limit <n>]]\n"
               "                      [--apply-from <retagged.db>] [--config <path>]\n"
               "       lilia segment [--terms | --count <n> | --bench | --pieces | --homophones] < lines.txt\n"
               "       lilia --version | --help\n",
               output);
}

void print_version() {
    std::printf("lilia %.*s (%.*s%s)\n", static_cast<int>(xmake::version::version.size()), xmake::version::version.data(),
                static_cast<int>(std::min<usize>(xmake::version::commit.size(), 12)), xmake::version::commit.data(),
                xmake::version::dirty ? "-dirty" : "");
}

log::Level parse_level(std::string_view name) {
    if (name == "debug") return log::Level::DEBUG;
    if (name == "warn") return log::Level::WARN;
    if (name == "error") return log::Level::ERROR;
    return log::Level::INFO;
}

Task<> serve(std::shared_ptr<onebot::Session> session, chat::Responder &responder) {
    auto ended = co_await session->run([&responder](onebot::Event event) { responder.on_event(std::move(event)); });
    responder.detach();
    if (ended.has_error()) log::warn("OneBot session ended: {}", ended.error().message());
}

Task<> identify(std::shared_ptr<onebot::Session> session, chat::Responder &responder) {
    auto login = co_await session->get_login_info();
    if (login.has_error()) {
        log::error("get_login_info failed: {}", login.error().message());
        session->close();
        co_return;
    }
    log::info("connected to NapCat as {} ({})", login->nickname, login->user_id);
    responder.attach(std::move(session), *std::move(login));
}

/// Keeps one OneBot session alive, reconnecting with capped backoff.
Task<> supervise(const Config &config, chat::Responder &responder) {
    auto backoff = 1s;
    for (;;) {
        const auto started = std::chrono::steady_clock::now();
        auto opened = co_await onebot::Session::open({
            .connect = {.host = config.onebot.host, .port = config.onebot.port, .path = config.onebot.path},
            .access_token = config.onebot.access_token,
        });
        if (opened.has_error()) {
            log::warn("cannot connect to OneBot at {}:{}: {}", config.onebot.host, config.onebot.port, opened.error().message());
        } else {
            auto session = *std::move(opened);
            lighter::TaskGroup connection(lighter::EventLoop::current());
            connection.spawn(serve(session, responder));
            connection.spawn(identify(session, responder));
            co_await connection.join();
            if (std::chrono::steady_clock::now() - started > 60s) backoff = 1s;
        }
        log::info("reconnecting in {}s", backoff.count());
        co_await lighter::sleep(backoff);
        backoff = std::min<std::chrono::seconds>(backoff * 2, 60s);
    }
}

/// Everything the current configuration embeds: memory, learned patterns and
/// the sticker pool.
std::vector<std::string> embedded_texts(const Config &config, store::Store &store) {
    auto texts = memory::embedded_texts(store);
    if (config.learning.enabled) std::ranges::move(learn::embedded_texts(store), std::back_inserter(texts));
    if (config.stickers.enabled) {
        if (auto pool = store.stickers("registered"); pool.has_value()) {
            for (const auto &sticker : *pool) {
                std::ranges::move(sticker::embedding_inputs(sticker, config.data_dir, config.memory.embed_images),
                                  std::back_inserter(texts));
            }
        }
    }
    return texts;
}

/// Embeds `texts` under the current profile (everything, after a model
/// change), then drops other profiles. Returns how many have no vector.
Task<usize> backfill(retrieval::Retriever &retriever, store::Store &store, std::vector<std::string> texts) {
    const auto started = std::chrono::steady_clock::now();
    const usize total = texts.size();
    const auto missing = co_await retriever.backfill(std::move(texts));
    const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started);
    log::info("retrieval: {} text(s) ready under {} in {} ms{}", total - missing, retriever.profile, took.count(),
              missing ? std::format(", {} not embedded", missing) : "");
    if (missing > 0) co_return missing;
    if (auto dropped = store.delete_embeddings_except(retriever.profile); dropped.has_error()) {
        log::error("retrieval: {}", dropped.error().message());
    } else if (*dropped > 0) {
        log::info("retrieval: dropped {} vector(s) of other models", *dropped);
    }
    co_return 0;
}

/// `lilia retrieval`: opens (and migrates) the database, backfills like a
/// start, then times a sticker search, a search of each allowed chat and an
/// expression search for each query. Prints scores and sticker tags only,
/// never chat text.
struct LabelQuery {
    std::string query;
    std::vector<i64> good;
    std::vector<i64> ok;
};

struct LabelFile {
    std::vector<LabelQuery> queries;
};

/// `lilia retrieval --labels`: scores sticker search against hand labels
/// (scripts/sticker_label_page.py): nDCG@k with gains good 2 / ok 1, recall@k
/// of the good ones, MRR and precision@1 of good-or-ok, fused and reranked.
Task<int> evaluate_stickers(const Config &config, retrieval::Retriever &retriever, const std::vector<retrieval::Document> &stickers,
                            const std::filesystem::path &labels_path) {
    std::ifstream file(labels_path, std::ios::binary);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    LabelFile labels;
    if (!file || glz::read<glz::opts{.null_terminated = false, .error_on_unknown_keys = false}>(labels, buffer.str())) {
        log::error("cannot read labels from {}", labels_path.string());
        co_return 1;
    }
    const usize k = std::max<usize>(config.stickers.candidates, 1);
    struct Score {
        double ndcg = 0, recall = 0, mrr = 0, p1 = 0, ms = 0;
    };
    const auto score = [&](const std::vector<retrieval::Match> &matches, const LabelQuery &label) {
        const auto gain = [&](i64 id) {
            return std::ranges::contains(label.good, id) ? 2.0 : std::ranges::contains(label.ok, id) ? 1.0 : 0.0;
        };
        Score out;
        double dcg = 0;
        usize good_found = 0;
        for (usize r = 0; r < matches.size() && r < k; ++r) {
            const auto id = stickers[matches[r].index].id;
            const double g = gain(id);
            dcg += g / std::log2(static_cast<double>(r) + 2);
            good_found += g == 2.0;
            if (g > 0 && out.mrr == 0) out.mrr = 1.0 / static_cast<double>(r + 1);
            if (r == 0) out.p1 = g > 0;
        }
        std::vector<double> ideal(label.good.size(), 2.0);
        ideal.insert(ideal.end(), label.ok.size(), 1.0);
        double idcg = 0;
        for (usize r = 0; r < ideal.size() && r < k; ++r) idcg += ideal[r] / std::log2(static_cast<double>(r) + 2);
        out.ndcg = idcg > 0 ? dcg / idcg : 0;
        out.recall = label.good.empty() ? 0 : static_cast<double>(good_found) / static_cast<double>(std::min(label.good.size(), k));
        return out;
    };
    const bool rerank = retriever.can_rerank() && config.memory.rerank_stickers > 0;
    Score fused_total, reranked_total;
    usize counted = 0;
    std::printf("%-12s %-30s %s\n", "query", "fused nDCG/recall/MRR/P@1", rerank ? "reranked nDCG/recall/MRR/P@1 (ms)" : "");
    for (const auto &label : labels.queries) {
        if (label.good.empty() && label.ok.empty()) continue;
        ++counted;
        auto started = std::chrono::steady_clock::now();
        const auto fused =
            score(co_await retriever.search(retrieval::k_sticker_task, label.query, stickers, {.top_k = k, .lexical = true}), label);
        std::string line = std::format("{:<12} {:.2f}/{:.2f}/{:.2f}/{:.0f}", label.query, fused.ndcg, fused.recall, fused.mrr, fused.p1);
        fused_total.ndcg += fused.ndcg, fused_total.recall += fused.recall, fused_total.mrr += fused.mrr, fused_total.p1 += fused.p1;
        if (rerank) {
            started = std::chrono::steady_clock::now();
            auto reranked = score(co_await retriever.search(retrieval::k_sticker_task, label.query, stickers,
                                                            {.top_k = k,
                                                             .lexical = true,
                                                             .rerank = config.memory.rerank_stickers,
                                                             .rerank_task = retrieval::k_sticker_rerank_task}),
                                  label);
            reranked.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            line += std::format("            {:.2f}/{:.2f}/{:.2f}/{:.0f} ({:.0f})", reranked.ndcg, reranked.recall, reranked.mrr,
                                reranked.p1, reranked.ms);
            reranked_total.ndcg += reranked.ndcg, reranked_total.recall += reranked.recall, reranked_total.mrr += reranked.mrr,
                reranked_total.p1 += reranked.p1, reranked_total.ms += reranked.ms;
        }
        std::printf("%s\n", line.c_str());
    }
    if (counted == 0) {
        std::printf("no labelled queries\n");
        co_return 1;
    }
    const auto n = static_cast<double>(counted);
    std::printf("mean over %zu queries, k=%zu: fused nDCG %.3f recall %.3f MRR %.3f P@1 %.3f\n", counted, k, fused_total.ndcg / n,
                fused_total.recall / n, fused_total.mrr / n, fused_total.p1 / n);
    if (rerank) {
        std::printf("reranked from %zu (pictures %s): nDCG %.3f recall %.3f MRR %.3f P@1 %.3f, %.0f ms per search\n",
                    config.memory.rerank_stickers, config.memory.rerank_images ? "on" : "off", reranked_total.ndcg / n,
                    reranked_total.recall / n, reranked_total.mrr / n, reranked_total.p1 / n, reranked_total.ms / n);
    }
    co_return 0;
}

/// `lilia stickers`: tags on a copy of the database, to score a tagging
/// change with `retrieval --labels` before it goes live, and copying the
/// winning tags over. Stickers a person edited are never touched.
struct StickerTool {
    bool retag = false;
    bool gifs = false;
    std::optional<bool> frames; ///< default: model.vision_gif_frames
    usize limit = 0;
    std::optional<std::filesystem::path> prompt;
    std::optional<std::filesystem::path> apply_from;
};

/// "无语 97 (49%)、无奈 34 (17%)…": the most common tags of the pool.
std::string top_tags(const std::vector<store::Sticker> &stickers, usize count = 12) {
    std::map<std::string, usize> tally;
    for (const auto &sticker : stickers) {
        std::string_view rest = sticker.tags;
        while (!rest.empty()) {
            const auto cut = rest.find("、");
            const auto tag = rest.substr(0, cut);
            if (!tag.empty()) ++tally[std::string(tag)];
            rest = cut == std::string_view::npos ? std::string_view{} : rest.substr(cut + std::string_view("、").size());
        }
    }
    std::vector<std::pair<std::string, usize>> sorted(tally.begin(), tally.end());
    std::ranges::stable_sort(sorted, [](const auto &a, const auto &b) { return a.second > b.second; });
    std::string out;
    for (usize i = 0; i < std::min(count, sorted.size()); ++i) {
        if (!out.empty()) out += "、";
        out += std::format("{} {} ({:.0f}%)", sorted[i].first, sorted[i].second,
                           100.0 * static_cast<double>(sorted[i].second) / static_cast<double>(std::max<usize>(1, stickers.size())));
    }
    return out;
}

Task<int> run_stickers(const Config &config, std::optional<std::filesystem::path> database, StickerTool tool) {
    if (!database) {
        log::error("lilia stickers needs --db <copy of lilia.db>; it never writes the live database by default");
        co_return 2;
    }
    auto store = store::Store::open(*database);
    if (store.has_error()) {
        log::error("{}", store.error().message());
        co_return 1;
    }
    const auto registered = [&] {
        auto pool = store->stickers("registered");
        return pool.has_value() ? *std::move(pool) : std::vector<store::Sticker>{};
    };
    std::printf("tags before: %s\n", top_tags(registered()).c_str());

    if (tool.apply_from) {
        auto source = store::Store::open_readonly(*tool.apply_from);
        if (source.has_error()) {
            log::error("{}", source.error().message());
            co_return 1;
        }
        usize copied = 0;
        usize kept = 0;
        for (const auto status : {"registered", "spare"}) {
            auto stickers = source->stickers(status);
            for (const auto &sticker : stickers.has_value() ? *stickers : std::vector<store::Sticker>{}) {
                auto target = store->sticker_by_hash(sticker.sha1);
                if (!target.has_value() || !*target) continue;
                const auto &current = **target;
                if (current.modified_by == "USER" || sticker.modified_by == "USER") {
                    ++kept;
                } else if (current.tags != sticker.tags || current.description != sticker.description) {
                    std::ignore = store->describe_sticker(current.id, sticker.description, sticker.tags, current.status);
                    ++copied;
                }
            }
        }
        std::printf("copied the tags of %zu stickers; %zu edited by a person kept\ntags after: %s\n", copied, kept,
                    top_tags(registered()).c_str());
        co_return 0;
    }
    if (!tool.retag) co_return 0;

    if (config.model.vision_model.empty()) {
        log::error("model.vision_model is empty: nothing to tag with");
        co_return 1;
    }
    sticker::Collector collector(config.stickers, config.data_dir, *store, [](Task<>) {});
    collector.vision = make_vision(config.model);
    collector.vision_model = config.model.vision_model;
    collector.frames = tool.frames.value_or(config.model.vision_gif_frames);
    if (tool.prompt) {
        std::ifstream file(*tool.prompt, std::ios::binary);
        std::stringstream text;
        text << file.rdbuf();
        if (!file || text.str().empty()) {
            log::error("cannot read {}", tool.prompt->string());
            co_return 1;
        }
        collector.prompt = text.str();
    }
    std::vector<store::Sticker> todo;
    for (const auto status : {"registered", "spare"}) {
        auto stickers = store->stickers(status);
        for (auto &sticker : stickers.has_value() ? *stickers : std::vector<store::Sticker>{}) {
            if (sticker.modified_by == "USER" || (tool.gifs && sticker.media_type != "image/gif")) continue;
            if (tool.limit > 0 && todo.size() >= tool.limit) break;
            todo.push_back(std::move(sticker));
        }
    }
    usize next = 0;
    usize tagged = 0;
    usize failed = 0;
    usize unfit = 0;
    // A few calls at a time; each takes seconds.
    auto worker = [&]() -> Task<> {
        while (next < todo.size()) {
            const auto sticker = todo[next++];
            const auto bytes = collector.read(sticker);
            const auto answer = bytes ? co_await collector.describe(sticker, *bytes, "") : std::nullopt;
            if (!answer) {
                ++failed;
                continue;
            }
            // The status stays: this measures tags, not the fitness verdict.
            if (!answer->usable) ++unfit;
            std::ignore = store->describe_sticker(sticker.id, answer->description, answer->tags, sticker.status);
            if (++tagged % 20 == 0) log::info("tagged {} of {}", tagged, todo.size());
        }
    };
    lighter::TaskGroup tasks(lighter::EventLoop::current());
    for (int i = 0; i < 4; ++i) tasks.spawn(worker());
    co_await tasks.join();
    std::printf("tagged %zu of %zu (%zu failed; %zu would be rejected)\ntags after: %s\n", tagged, todo.size(), failed, unfit,
                top_tags(registered()).c_str());
    co_return failed == todo.size() && !todo.empty() ? 1 : 0;
}

Task<int> run_retrieval(const Config &config, std::filesystem::path database, std::vector<std::string> queries,
                        std::optional<std::filesystem::path> labels) {
    auto store = store::Store::open(database);
    if (store.has_error()) {
        log::error("{}", store.error().message());
        co_return 1;
    }
    retrieval::Retriever retriever(config.memory, *store);
    memory::Memory memory(config.memory, config.chat.name, config.chat.utc_offset_minutes, *store, retriever);
    if (co_await backfill(retriever, *store, embedded_texts(config, *store)) > 0) co_return 1;
    const auto time = [](std::chrono::steady_clock::time_point started) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    };
    auto pool = store->stickers("registered");
    std::vector<retrieval::Document> stickers;
    for (const auto &sticker : pool.has_value() ? *pool : std::vector<store::Sticker>{}) {
        stickers.push_back({.id = sticker.id,
                            .texts = sticker::embedding_inputs(sticker, config.data_dir, config.memory.embed_images),
                            .lexical = sticker.tags + " " + sticker.description});
    }
    if (labels) co_return co_await evaluate_stickers(config, retriever, stickers, *labels);
    std::vector<retrieval::Document> expressions;
    for (auto &text : learn::embedded_texts(*store)) expressions.push_back({.texts = {std::move(text)}});
    for (const auto &query : queries) {
        std::printf("== %s\n", query.c_str());
        for (int round = 0; round < 2; ++round) {
            const auto started = std::chrono::steady_clock::now();
            const auto matches = co_await retriever.search(retrieval::k_sticker_task, query, stickers, {.top_k = 8, .lexical = true});
            std::printf("stickers (%s): %.1f ms\n", round == 0 ? "new query" : "same query again", time(started));
            if (round == 1) continue;
            for (const auto &match : matches) {
                const auto &sticker = (*pool)[match.index];
                std::printf("  s%lld %.3f [%s]\n", static_cast<long long>(sticker.id), match.similarity.value_or(0), sticker.tags.c_str());
            }
        }
        if (retriever.can_rerank() && config.memory.rerank_stickers > 0) {
            const auto started = std::chrono::steady_clock::now();
            const auto matches = co_await retriever.search(
                retrieval::k_sticker_task, query, stickers,
                {.top_k = 8, .lexical = true, .rerank = config.memory.rerank_stickers, .rerank_task = retrieval::k_sticker_rerank_task});
            std::printf("stickers reranked from %zu: %.1f ms\n", config.memory.rerank_stickers, time(started));
            for (const auto &match : matches) {
                const auto &sticker = (*pool)[match.index];
                std::printf("  s%lld %.3f (dense %.3f) [%s]\n", static_cast<long long>(sticker.id), match.relevance.value_or(0),
                            match.similarity.value_or(0), sticker.tags.c_str());
            }
        }
        for (const auto group : config.chat.groups) {
            const auto started = std::chrono::steady_clock::now();
            const auto hits = co_await memory.search({.type = "group", .id = group}, query);
            std::printf("chat %lld: %zu hit(s)%s, %.1f ms\n", static_cast<long long>(group), hits.size(),
                        hits.empty() ? "" : std::format(", best {:.3f}", hits.front().score).c_str(), time(started));
        }
        const auto started = std::chrono::steady_clock::now();
        const auto picked = co_await retriever.search(retrieval::k_expression_task, query, expressions, {.top_k = 30, .mmr_lambda = 0.85});
        std::printf("expressions: %zu of %zu, best %.3f, %.1f ms\n", picked.size(), expressions.size(),
                    picked.empty() ? 0.0 : picked.front().similarity.value_or(0), time(started));
    }
    co_return 0;
}

Task<int> run_app(const Config &config) {
    const std::filesystem::path data_dir = config.data_dir;
    auto store = store::Store::open(data_dir / "lilia.db");
    if (store.has_error()) {
        log::error("{}", store.error().message());
        co_return 1;
    }

    lighter::TaskGroup tasks(lighter::EventLoop::current());
    chat::Responder responder(config, *store, make_planner(config.model), make_searcher(config.model), make_writer(config.model),
                              data_dir / "persona.md", [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    responder.writer_fallbacks = make_writer_fallbacks(config.model);
    responder.planner_fallbacks = make_planner_fallbacks(config.model);
    archive::Archiver archiver(config.archive, data_dir, *store);
    if (!config.model.vision_model.empty()) {
        archiver.vision = make_vision(config.model);
        archiver.vision_model = config.model.vision_model;
        archiver.gif_frames = config.model.vision_gif_frames;
    }
    // Also with archiving off: forwards sent by id are still fetched.
    responder.archiver = &archiver;
    learn::Learner learner(config.learning, config.chat.name, config.chat.aliases, *store, make_learner(config.model),
                           [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    if (config.learning.enabled) responder.learner = &learner;
    learner.frequency_chats = config.chat.groups;
    sticker::Collector collector(config.stickers, data_dir, *store, [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    if (config.stickers.enabled) {
        if (!config.model.vision_model.empty()) {
            collector.vision = make_vision(config.model);
            collector.vision_model = config.model.vision_model;
            collector.frames = config.model.vision_gif_frames;
        }
        collector.background_budget = config.learning.daily_token_budget;
        responder.stickers = &collector;
    }
    retrieval::Retriever retriever(config.memory, *store);
    memory::Memory memory(config.memory, config.chat.name, config.chat.utc_offset_minutes, *store, retriever);
    if (config.memory.enabled) {
        responder.memory = &memory;
        memory.summarizer = make_learner(config.model);
        memory.summarizer_model = config.model.helper();
        learner.retriever = &retriever;
        tasks.spawn([](retrieval::Retriever &retriever, store::Store &store, std::vector<std::string> texts) -> Task<> {
            std::ignore = co_await backfill(retriever, store, std::move(texts));
        }(retriever, *store, embedded_texts(config, *store)));
        for (const auto group : config.chat.groups) tasks.spawn(memory.index({.type = "group", .id = group}));
        for (const auto user : config.chat.private_users) tasks.spawn(memory.index({.type = "private", .id = user}));
    }
    if (auto loaded = responder.reload_persona(); loaded.has_error()) {
        log::error("{}", loaded.error().message());
        co_return 1;
    }

    log::info("lilia {} starting; data in {}; {} allowed group(s); writer {}, planner {}", xmake::version::version, data_dir.string(),
              config.chat.groups.size(), config.model.model, config.model.planner_model);
    tasks.spawn(supervise(config, responder));
    co_await tasks.join();
    co_return 0;
}

} // namespace

int main(int argc, char **argv) {
    std::optional<std::filesystem::path> config_path;
    std::optional<std::filesystem::path> replay_log;
    std::optional<std::filesystem::path> persona_path;
    std::optional<std::filesystem::path> habits_db;
    std::optional<std::filesystem::path> writer_prompt;
    std::optional<std::filesystem::path> stickers_db;
    std::optional<usize> continue_at;
    std::optional<std::filesystem::path> learn_log;
    bool learn_history = false;
    std::vector<i64> history_groups;
    usize history_count = 300;
    bool list_expressions = false;
    bool jargon = false;
    Manage manage;
    bool review = false;
    bool apply = false;
    std::vector<std::string> keep;
    usize limit = 100;
    bool check_only = false;
    bool retrieval_check = false;
    std::vector<std::string> queries;
    std::optional<std::filesystem::path> database_path;
    std::optional<std::filesystem::path> labels_path;
    bool sticker_tool = false;
    StickerTool stickers;
    if (argc >= 2 && std::string_view(argv[1]) == "segment") {
        // Debugging aids, one stdin line per message: jieba words of each
        // line joined by "/"; --terms: its meaningful terms; --count <n>: the
        // frequent-term tallies of every n lines ("term<TAB>count<TAB>messages").
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        const auto batch = static_cast<usize>(argc >= 4 ? std::max(1, std::atoi(argv[3])) : 30);
        const auto join = [](const auto &words) {
            std::string joined;
            for (const auto &word : words) {
                if (!joined.empty()) joined += '/';
                joined += word;
            }
            return joined;
        };
        std::vector<std::string> lines;
        const auto flush = [&] {
            for (const auto &tally : learn::count_terms(lines))
                std::printf("%s\t%lld\t%lld\n", tally.term.c_str(), static_cast<long long>(tally.count),
                            static_cast<long long>(tally.message_count));
            std::printf("--\n");
            lines.clear();
        };
        std::string line;
        if (mode == "--bench") {
            // Timings without output: loading, cutting, terms, 30-line tallies.
            std::vector<std::string> all;
            usize bytes = 0;
            while (std::getline(std::cin, line)) {
                bytes += line.size() + 1;
                all.push_back(line);
            }
            const auto time = [](auto &&work) {
                const auto started = std::chrono::steady_clock::now();
                work();
                return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            };
            usize tokens = 0;
            const double load = time([] { learn::cut("好"); });
            const double cut = time([&] {
                for (const auto &text : all) tokens += learn::cut(text).size();
            });
            const double terms = time([&] {
                for (const auto &text : all) learn::meaningful_terms(text);
            });
            const double tallies = time([&] {
                for (usize start = 0; start < all.size(); start += 30)
                    learn::count_terms({all.begin() + static_cast<std::ptrdiff_t>(start),
                                        all.begin() + static_cast<std::ptrdiff_t>(std::min(all.size(), start + 30))});
            });
            std::fprintf(stderr, "%zu lines, %.1f MB, %zu tokens\nload %.0f ms\ncut %.0f ms (%.1f MB/s)\nterms %.0f ms\ntallies %.0f ms\n",
                         all.size(), static_cast<double>(bytes) / 1e6, tokens, load, cut, static_cast<double>(bytes) / 1e3 / cut, terms,
                         tallies);
            return 0;
        }
        while (std::getline(std::cin, line)) {
            if (mode == "--pieces") {
                // Reply pieces before merging: text \x1f separator, pieces split by \x1e.
                std::string out;
                for (const auto &piece : chat::split_pieces(line)) out += piece.text + "\x1f" + piece.separator + "\x1e";
                std::printf("%s\n", out.c_str());
            } else if (mode == "--homophones") {
                // Typo candidates for a character or a word (min_freq 9, no tone errors).
                const auto chars = learn::utf8_chars(line);
                std::vector<std::string> out;
                if (chars.size() == 1) {
                    for (const auto c : chat::similar_chars(chars[0].cp, chat::reading_of(chars[0].cp), "", 9))
                        out.push_back(learn::encode_utf8(c));
                } else {
                    out = chat::word_homophones(line, 9);
                }
                std::printf("%s\n", join(out).c_str());
            } else if (mode == "--terms") {
                std::printf("%s\n", join(learn::meaningful_terms(line)).c_str());
            } else if (mode == "--count") {
                lines.push_back(line);
                if (lines.size() == batch) flush();
            } else {
                std::printf("%s\n", join(learn::cut(line)).c_str());
            }
        }
        if (!lines.empty()) flush();
        return 0;
    }
    int first = 1;
    if (argc > 2 && std::string_view(argv[1]) == "replay") {
        replay_log = argv[2];
        first = 3;
    } else if (argc > 2 && std::string_view(argv[1]) == "learn") {
        learn_log = argv[2];
        first = 3;
    } else if (argc > 1 && std::string_view(argv[1]) == "learn-history") {
        learn_history = true;
        first = 2;
    } else if (argc > 1 && std::string_view(argv[1]) == "expressions") {
        list_expressions = true;
        first = 2;
    } else if (argc > 1 && std::string_view(argv[1]) == "retrieval") {
        retrieval_check = true;
        first = 2;
    } else if (argc > 1 && std::string_view(argv[1]) == "stickers") {
        sticker_tool = true;
        first = 2;
    } else if (argc > 1 && std::string_view(argv[1]) == "jargon") {
        jargon = true;
        manage.action = "list";
        first = 2;
    }
    for (int i = first; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(stdout);
            return 0;
        }
        if (arg == "--version") {
            print_version();
            return 0;
        }
        if (arg == "--check-config") {
            check_only = true;
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--persona" && replay_log && i + 1 < argc) {
            persona_path = argv[++i];
        } else if (arg == "--habits" && replay_log && i + 1 < argc) {
            habits_db = argv[++i];
        } else if (arg == "--writer-prompt" && replay_log && i + 1 < argc) {
            writer_prompt = argv[++i];
        } else if (arg == "--stickers" && replay_log && i + 1 < argc) {
            stickers_db = argv[++i];
        } else if (arg == "--continue-at" && replay_log && i + 1 < argc) {
            continue_at = static_cast<usize>(std::max(0, std::atoi(argv[++i])));
        } else if (arg == "--query" && retrieval_check && i + 1 < argc) {
            queries.emplace_back(argv[++i]);
        } else if (arg == "--db" && (retrieval_check || sticker_tool) && i + 1 < argc) {
            database_path = argv[++i];
        } else if (arg == "--labels" && retrieval_check && i + 1 < argc) {
            labels_path = argv[++i];
        } else if (arg == "--retag" && sticker_tool) {
            stickers.retag = true;
        } else if (arg == "--gifs" && sticker_tool) {
            stickers.gifs = true;
        } else if ((arg == "--frames" || arg == "--no-frames") && sticker_tool) {
            stickers.frames = arg == "--frames";
        } else if (arg == "--prompt" && sticker_tool && i + 1 < argc) {
            stickers.prompt = argv[++i];
        } else if (arg == "--apply-from" && sticker_tool && i + 1 < argc) {
            stickers.apply_from = argv[++i];
        } else if (arg == "--limit" && sticker_tool && i + 1 < argc) {
            stickers.limit = static_cast<usize>(std::max(1, std::atoi(argv[++i])));
        } else if (arg == "--group" && learn_history && i + 1 < argc) {
            history_groups.push_back(std::atoll(argv[++i]));
        } else if (arg == "--count" && learn_history && i + 1 < argc) {
            history_count = static_cast<usize>(std::max(1, std::atoi(argv[++i])));
        } else if (arg == "--review" && list_expressions) {
            review = true;
        } else if (arg == "--apply" && list_expressions) {
            apply = true;
        } else if (arg == "--keep" && list_expressions && i + 1 < argc) {
            keep.emplace_back(argv[++i]);
        } else if (arg == "--limit" && (list_expressions || jargon) && i + 1 < argc) {
            limit = static_cast<usize>(std::max(1, std::atoi(argv[++i])));
            manage.limit = limit;
        } else if (arg == "--rejected" && list_expressions) {
            manage.action = "rejected";
        } else if ((arg == "--rescue" || arg == "--check" || arg == "--uncheck") && list_expressions && i + 1 < argc) {
            manage.action = std::string(arg.substr(2));
            manage.id = std::atoll(argv[++i]);
        } else if (arg == "--set" && jargon && i + 2 < argc) {
            manage.action = "set";
            manage.term = argv[++i];
            manage.meaning = argv[++i];
        } else if (arg == "--delete" && jargon && i + 1 < argc) {
            manage.action = "delete";
            manage.term = argv[++i];
        } else {
            print_usage(stderr);
            return 2;
        }
    }

    const auto path = config_path.value_or(default_config_path());
    // Listing learned expressions reads the database only; no keys needed.
    auto config = load_config(path, !(list_expressions || jargon || (sticker_tool && !stickers.retag)) || review);
    if (config.has_error()) {
        log::error("{}: {}", path.string(), config.error().message());
        return 1;
    }
    log::g_threshold = parse_level(config->log_level);
    if (check_only) {
        log::info("{} is valid", path.string());
        return 0;
    }
    const auto database = std::filesystem::path(config->data_dir) / "lilia.db";
    if (jargon) return manage_jargon(database, manage);
    if (list_expressions && !manage.action.empty()) return manage_expressions(database, manage);
    if (list_expressions && !review) return print_expressions(database, limit);

    lighter::EventLoop loop;
    auto interrupts = lighter::InterruptSource::create(loop);
    if (!interrupts) {
        log::error("cannot watch process signals: {}", interrupts.error().message());
        return 1;
    }
    auto app = lighter::with_token(replay_log ?
                                       run_replay(*config, *replay_log, persona_path, habits_db, writer_prompt, stickers_db, continue_at) :
                                   learn_log       ? run_learn(*config, *learn_log) :
                                   learn_history   ? run_learn_history(*config, history_groups, history_count) :
                                   review          ? review_expressions(*config, apply, keep) :
                                   retrieval_check ? run_retrieval(*config, database_path.value_or(database), queries, labels_path) :
                                   sticker_tool    ? run_stickers(*config, database_path, stickers) :
                                                     run_app(*config),
                                   interrupts->token());
    loop.schedule(app);
    loop.run();

    auto result = app.result();
    if (result.is_cancelled()) {
        log::info("stopped");
        return 0;
    }
    return *result;
}
