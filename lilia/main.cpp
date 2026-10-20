#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <span>
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
#include <lilia/chat/responder.h>
#include <lilia/config.h>
#include <lilia/learn/learner.h>
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
               "                    [--writer-prompt <writer.md>] [--stickers <lilia.db>] [--continue-at <line> [--free]]\n"
               "       lilia replay <probes.jsonl> --probe [--repeat <n>] [--planner-prompt <planner.md>] [--config <path>]\n"
               "       lilia replay --probe-db <copy.db> [--repeat <n>] [--limit <n>] [--planner-prompt <planner.md>] [--config <path>]\n"
               "       lilia learn <chat.jsonl> [--config <path>]\n"
               "       lilia learn-history [--group <id>]... [--count <n>] [--config <path>]\n"
               "       lilia expressions [--config <path>] [--limit <n>]\n"
               "       lilia expressions --review [--apply] [--keep <text>]... [--config <path>]\n"
               "       lilia expressions --rejected | --rescue <review id> | --check <id> | --uncheck <id>\n"
               "       lilia jargon [--limit <n>] | --set <term> <meaning> | --delete <term>\n"
               "       lilia retrieval [--query <text>]... [--labels <labels.json>] [--db <lilia.db>] [--config <path>]\n"
               "       lilia stickers --db <copy.db> [--retag [--gifs] [--prompt <sticker.md>] [--limit <n>]]\n"
               "                      [--apply-from <retagged.db>] [--learn-usage] [--forget-group <id>] [--config <path>]\n"
               "       lilia feedback [--backfill [--days <n>]] [--eval] [--report [--days <n>]] [--import-manual <labels.json>]\n"
               "                      [--db <lilia.db>] [--config <path>]\n"
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
                std::ranges::move(sticker::embedding_inputs(sticker), std::back_inserter(texts));
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
/// of the good ones, MRR and precision@1 of good-or-ok.
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
        double ndcg = 0, recall = 0, mrr = 0, p1 = 0;
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
    Score total;
    usize counted = 0;
    std::printf("%-12s %s\n", "query", "nDCG/recall/MRR/P@1");
    for (const auto &label : labels.queries) {
        if (label.good.empty() && label.ok.empty()) continue;
        ++counted;
        const auto found =
            score(co_await retriever.search(retrieval::k_sticker_task, label.query, stickers, {.top_k = k, .lexical = true}), label);
        std::printf("%-12s %.2f/%.2f/%.2f/%.0f\n", label.query.c_str(), found.ndcg, found.recall, found.mrr, found.p1);
        total.ndcg += found.ndcg, total.recall += found.recall, total.mrr += found.mrr, total.p1 += found.p1;
    }
    if (counted == 0) {
        std::printf("no labelled queries\n");
        co_return 1;
    }
    const auto n = static_cast<double>(counted);
    std::printf("mean over %zu queries, k=%zu: nDCG %.3f recall %.3f MRR %.3f P@1 %.3f\n", counted, k, total.ndcg / n, total.recall / n,
                total.mrr / n, total.p1 / n);
    co_return 0;
}

/// `lilia stickers`: tags on a copy of the database, to score a tagging
/// change with `retrieval --labels` before it goes live, and copying the
/// winning tags over. Stickers a person edited are never touched.
struct StickerTool {
    bool retag = false;
    bool gifs = false;
    usize limit = 0;
    std::optional<std::filesystem::path> prompt;
    std::optional<std::filesystem::path> apply_from;
    /// Infers the usage of every registered sticker with enough snippets.
    bool learn_usage = false;
    /// Forgets one group's usage snippets.
    std::optional<i64> forget_group;
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
    if (tool.forget_group) {
        auto forgotten = store->delete_sticker_evidence_of_group(*tool.forget_group);
        if (forgotten.has_error()) {
            log::error("{}", forgotten.error().message());
            co_return 1;
        }
        std::printf("forgot %lld snippets of group %lld\n", static_cast<long long>(*forgotten), static_cast<long long>(*tool.forget_group));
        co_return 0;
    }
    if (tool.learn_usage) {
        sticker::Collector collector(config.stickers, config.data_dir, *store, [](Task<>) {});
        collector.helper = make_learner(config.model);
        collector.helper_model = config.model.helper();
        collector.self_name = config.chat.name;
        usize learned = 0;
        for (const auto &sticker : registered()) {
            if (sticker.usage_count < 3) continue;
            co_await collector.infer_usage(sticker);
            ++learned;
        }
        std::printf("inferred the usage of %zu stickers\n", learned);
        co_return 0;
    }
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

/// `lilia feedback`: hindsight judging of stored history, the addressee
/// models' evaluation, a report of how her replies went down, and hand
/// labels. Works on the given database.
struct FeedbackTool {
    bool backfill = false;
    i64 days = 14;
    bool eval = false;
    bool report = false;
    std::optional<std::filesystem::path> import_manual;
};

double auc(const std::vector<std::pair<double, double>> &scored) {
    // Probability that a random positive outranks a random negative.
    double positives = 0;
    double negatives = 0;
    double wins = 0;
    for (const auto &[p, label] : scored) (label > 0.5 ? positives : negatives) += 1;
    for (const auto &[p, label] : scored) {
        if (label < 0.5) continue;
        for (const auto &[q, other] : scored) {
            if (other > 0.5) continue;
            wins += p > q ? 1 : p == q ? 0.5 : 0;
        }
    }
    return positives > 0 && negatives > 0 ? wins / (positives * negatives) : 0;
}

Task<int> run_feedback(const Config &config, std::optional<std::filesystem::path> explicit_database, std::filesystem::path live,
                       FeedbackTool tool) {
    if ((tool.backfill || tool.import_manual) && !explicit_database) {
        log::error("lilia feedback --backfill/--import-manual write: name the database with --db");
        co_return 2;
    }
    auto store = store::Store::open(explicit_database.value_or(live));
    if (store.has_error()) {
        log::error("{}", store.error().message());
        co_return 1;
    }
    const i64 now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    lighter::TaskGroup tasks(lighter::EventLoop::current());
    feedback::Feedback feedback(config.feedback, config.chat, *store, make_judge(config.model),
                                [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    feedback.judge_model_name = config.model.helper();
    feedback.load();
    const i64 self_id = self_id_of(*store, config.chat.groups);

    if (tool.import_manual) {
        std::ifstream file(*tool.import_manual, std::ios::binary);
        std::stringstream text;
        text << file.rdbuf();
        struct ManualLabel {
            i64 chat_id = 0;
            i64 message_id = 0;
            std::string to;      ///< her, other, group, self or unclear (skipped)
            bool to_her = false; ///< older pages: only her or not
        };
        struct ManualLabels {
            std::vector<ManualLabel> labels;
        } manual;
        if (!file || glz::read<glz::opts{.null_terminated = false, .error_on_unknown_keys = false}>(manual, text.str())) {
            log::error("cannot read labels from {}", tool.import_manual->string());
            co_return 1;
        }
        // The judge's labels first: hand labels replace them.
        std::map<std::pair<i64, i64>, store::AddresseeLabel> judged;
        if (auto existing = store->addressee_labels(0, "judge"); existing.has_value()) {
            for (auto &label : *existing) judged[{label.chat.id, label.message_id}] = std::move(label);
        }
        usize stored = 0;
        usize compared = 0;
        usize agreed = 0;
        usize compared_kinds = 0;
        usize agreed_kinds = 0;
        for (const auto &label : manual.labels) {
            const store::ChatKey chat{.type = "group", .id = label.chat_id};
            if (label.to == "unclear") continue;
            const bool to_her = label.to.empty() ? label.to_her : label.to == "her";
            if (const auto it = judged.find({label.chat_id, label.message_id}); it != judged.end()) {
                ++compared;
                agreed += it->second.label == to_her;
                if (!label.to.empty() && !it->second.addressee.empty()) {
                    ++compared_kinds;
                    agreed_kinds += it->second.addressee == label.to;
                }
            }
            auto message = store->message(chat, label.message_id);
            if (!message.has_value() || !*message) continue;
            auto rows = store->messages_before(chat, (*message)->row_id, 40);
            if (!rows.has_value()) continue;
            rows->push_back(**message);
            const auto lookup = [&](i64 id) -> std::optional<store::StoredMessage> {
                auto found = store->message(chat, id);
                return found.has_value() && *found ? std::optional<store::StoredMessage>(**found) : std::nullopt;
            };
            const auto name_of = [&](i64 user) {
                auto name = store->sender_name(chat, user);
                return name.has_value() ? *name : std::string();
            };
            std::vector<chat::LineEdges> edges;
            for (const auto &row : *rows)
                edges.push_back(chat::line_edges(row, self_id, config.chat.name, config.chat.aliases, lookup, name_of));
            const auto x = chat::line_features(*rows, edges, rows->size() - 1);
            const std::vector<double> values(x.begin(), x.end());
            std::ignore = store->add_addressee_label({.chat = chat,
                                                      .message_id = label.message_id,
                                                      .row_id = (*message)->row_id,
                                                      .time = (*message)->time,
                                                      .feature_version = chat::k_feature_version,
                                                      .features = glz::write_json(values).value_or("[]"),
                                                      .label = to_her,
                                                      .source = "manual",
                                                      .prior = feedback::predict(feedback.model_for(chat), x),
                                                      .addressee = label.to});
            ++stored;
        }
        std::printf("stored %zu hand labels; the judge had labelled %zu of them and agreed on %zu (her or not)", stored, compared, agreed);
        if (compared_kinds > 0) std::printf(", on whom %zu of %zu", agreed_kinds, compared_kinds);
        std::printf("\n");
        co_return 0;
    }

    if (tool.backfill) {
        if (self_id == 0) log::warn("her QQ id is unknown (she never spoke): lines that @ her are not recognised");
        for (const auto group : config.chat.groups) {
            const store::ChatKey chat{.type = "group", .id = group};
            // Start where the period begins, unless judging already got further.
            auto through = store->feedback_progress(chat);
            i64 start = through.has_value() ? *through : 0;
            i64 cursor = 0;
            for (;;) {
                auto page = store->messages_after(chat, cursor, 500);
                if (!page.has_value() || page->empty()) break;
                const auto first_recent =
                    std::ranges::find_if(*page, [&](const auto &row) { return row.time >= now - tool.days * 86'400; });
                if (first_recent != page->end()) {
                    start = std::max(start, first_recent == page->begin() ? cursor : (first_recent - 1)->row_id);
                    break;
                }
                cursor = page->back().row_id;
            }
            std::ignore = store->set_feedback_progress(chat, start);
            usize windows = 0;
            while (co_await feedback.judge(chat, self_id, std::max<usize>(config.feedback.judge_window_lines, 1))) ++windows;
            std::printf("group %lld: %zu windows judged\n", static_cast<long long>(group), windows);
        }
        feedback.refit();
    }

    if (tool.eval) {
        auto labels = store->addressee_labels(chat::k_feature_version);
        if (!labels.has_value() || labels->size() < 10) {
            std::printf("too few labels (%zu)\n", labels.has_value() ? labels->size() : 0);
            co_return 0;
        }
        const auto samples = feedback::samples_of(*labels);
        // Earlier labels train, later ones test; fitted as a refit would,
        // gates and all, and once more without the gates.
        const usize cut = samples.size() * 7 / 10;
        const std::span<const feedback::LabelledSample> train(samples.data(), cut);
        const auto production = feedback::fit_models(train, config.feedback);
        auto no_gates = config.feedback;
        no_gates.min_rows_global = no_gates.min_rows_group = no_gates.min_positives = 0;
        const auto ungated = feedback::fit_models(train, no_gates);
        const auto bootstrap = feedback::bootstrap_weights();
        const auto score = [&](std::string_view name, const auto &weights_for) {
            std::vector<feedback::Sample> test;
            std::vector<std::pair<double, double>> scored;
            double brier = 0;
            for (usize i = cut; i < samples.size(); ++i) {
                const auto &weights = weights_for(samples[i].first);
                const double p = feedback::predict(weights, samples[i].second.x);
                scored.emplace_back(p, samples[i].second.label);
                brier += (p - samples[i].second.label) * (p - samples[i].second.label);
                test.push_back(samples[i].second);
            }
            double loss = 0;
            for (usize i = 0; i < test.size(); ++i) {
                const double p = std::clamp(scored[i].first, 1e-9, 1 - 1e-9);
                loss -= test[i].label * std::log(p) + (1 - test[i].label) * std::log(1 - p);
            }
            std::printf("%-18s log loss %.3f  Brier %.3f  AUC %.3f\n", std::string(name).c_str(), loss / static_cast<double>(test.size()),
                        brier / static_cast<double>(test.size()), auc(scored));
        };
        usize positives = 0;
        for (const auto &[chat, sample] : samples) positives += sample.label > 0.5;
        std::printf("%zu labels (%zu to her); train %zu, test %zu\n", samples.size(), positives, cut, samples.size() - cut);
        const auto gated = [](const feedback::FittedModel &model) { return model.fitted ? "fitted" : "not fitted"; };
        std::printf("on the training labels, a refit (min_rows_global %zu, min_rows_group %zu, min_positives %zu) would leave:\n",
                    config.feedback.min_rows_global, config.feedback.min_rows_group, config.feedback.min_positives);
        std::printf("  global: %s (%zu rows, %zu to her)\n", gated(production.global), production.global.rows, production.global.positives);
        for (const auto &[chat, model] : production.groups) {
            std::printf("  %s %lld: %s (%zu rows, %zu to her)\n", chat.type.c_str(), static_cast<long long>(chat.id), gated(model),
                        model.rows, model.positives);
        }
        score("bootstrap", [&](const store::ChatKey &) -> const feedback::Weights & { return bootstrap; });
        // What she would use: a group's own model, else the global one, else bootstrap.
        score("production", [&](const store::ChatKey &chat) -> const feedback::Weights & { return production.model_for(chat); });
        score("ungated global", [&](const store::ChatKey &) -> const feedback::Weights & { return ungated.global.weights; });
        score("ungated per group", [&](const store::ChatKey &chat) -> const feedback::Weights & { return ungated.model_for(chat); });
    }

    if (tool.report) {
        for (const auto group : config.chat.groups) {
            const store::ChatKey chat{.type = "group", .id = group};
            auto effects = store->reply_effects(chat, now - tool.days * 86'400);
            if (!effects.has_value() || effects->empty()) continue;
            usize misreads = 0;
            usize intrusions = 0;
            double reward = 0;
            std::map<std::string, usize> stances;
            for (const auto &effect : *effects) {
                misreads += effect.for_her == "no" && (effect.to_her == "called" || effect.to_her == "likely");
                intrusions += effect.contribution == "intrusion" || effect.contribution == "wrong_push";
                reward += effect.reward;
                ++stances[effect.stance];
            }
            std::string mix;
            for (const auto &[stance, count] : stances) mix += std::format(" {} {}", stance, count);
            std::printf("group %lld: %zu messages judged; misread %zu, intrusion or wrong push %zu, mean reward %.2f;%s\n",
                        static_cast<long long>(group), effects->size(), misreads, intrusions, reward / static_cast<double>(effects->size()),
                        mix.c_str());
        }
    }
    co_await tasks.join();
    co_return 0;
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
        stickers.push_back({.id = sticker.id, .texts = sticker::embedding_inputs(sticker), .lexical = sticker::lexical_text(sticker)});
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
    // The planner's decision log keeps half a year.
    constexpr i64 k_planner_turn_seconds = 180 * 86'400;
    std::ignore = store->prune_planner_turns(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count() -
        k_planner_turn_seconds);

    lighter::TaskGroup tasks(lighter::EventLoop::current());
    chat::Responder responder(config, *store, make_planner(config.model), make_searcher(config.model), make_writer(config.model),
                              data_dir / "persona.md", [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    responder.writer_fallbacks = make_writer_fallbacks(config.model);
    responder.planner_fallbacks = make_planner_fallbacks(config.model);
    responder.selector = make_selector(config.model);
    // Only Claude needs asking to cache, and only its cache needs keeping.
    if (config.model.cache_keepalive_minutes > 0 && config.model.planner_model.starts_with("claude"))
        responder.warmer = make_warmer(config.model);
    archive::Archiver archiver(config.archive, data_dir, *store);
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
        }
        collector.helper = make_learner(config.model);
        collector.helper_model = config.model.helper();
        collector.self_name = config.chat.name;
        collector.background_budget = config.learning.daily_token_budget;
        responder.stickers = &collector;
    }
    feedback::Feedback feedback(config.feedback, config.chat, *store, make_judge(config.model),
                                [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    // Always attached: the addressee prior needs no model calls; judging (and
    // so fitting) only runs with feedback.enabled.
    feedback.judge_model_name = config.model.helper();
    feedback.background_budget = config.learning.daily_token_budget;
    feedback.load();
    responder.feedback = &feedback;
    retrieval::Retriever retriever(config.memory, *store);
    memory::Memory memory(config.memory, config.chat.name, config.chat.utc_offset_minutes, *store, retriever);
    if (config.memory.enabled) {
        responder.memory = &memory;
        memory.summarizer = make_summarizer(config.model);
        memory.summarizer_model = config.model.helper();
        memory.data_dir = config.data_dir;
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
    bool replay = false;
    std::optional<std::filesystem::path> replay_log;
    std::optional<std::filesystem::path> persona_path;
    std::optional<std::filesystem::path> habits_db;
    std::optional<std::filesystem::path> writer_prompt;
    std::optional<std::filesystem::path> stickers_db;
    std::optional<usize> continue_at;
    bool free_turn = false;
    bool probe = false;
    std::optional<std::filesystem::path> planner_prompt;
    usize probe_repeat = 1;
    std::optional<std::filesystem::path> probe_db;
    usize probe_limit = 0;
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
    bool feedback_tool = false;
    FeedbackTool feedback_options;
    int first = 1;
    if (argc > 1 && std::string_view(argv[1]) == "replay") {
        // `--probe-db` probes stored chat instead of a log.
        replay = true;
        first = 2;
        if (argc > 2 && !std::string_view(argv[2]).starts_with('-')) {
            replay_log = argv[2];
            first = 3;
        }
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
    } else if (argc > 1 && std::string_view(argv[1]) == "feedback") {
        feedback_tool = true;
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
        } else if (arg == "--persona" && replay && i + 1 < argc) {
            persona_path = argv[++i];
        } else if (arg == "--habits" && replay_log && i + 1 < argc) {
            habits_db = argv[++i];
        } else if (arg == "--writer-prompt" && replay_log && i + 1 < argc) {
            writer_prompt = argv[++i];
        } else if (arg == "--stickers" && replay_log && i + 1 < argc) {
            stickers_db = argv[++i];
        } else if (arg == "--probe" && replay_log) {
            probe = true;
        } else if (arg == "--probe-db" && replay && !replay_log && i + 1 < argc) {
            probe_db = argv[++i];
        } else if (arg == "--limit" && replay && !replay_log && i + 1 < argc) {
            probe_limit = static_cast<usize>(std::max(1, std::atoi(argv[++i])));
        } else if (arg == "--planner-prompt" && replay && i + 1 < argc) {
            planner_prompt = argv[++i];
        } else if (arg == "--repeat" && replay && i + 1 < argc) {
            probe_repeat = static_cast<usize>(std::max(1, std::atoi(argv[++i])));
        } else if (arg == "--continue-at" && replay_log && i + 1 < argc) {
            continue_at = static_cast<usize>(std::max(0, std::atoi(argv[++i])));
        } else if (arg == "--free" && replay_log) {
            free_turn = true;
        } else if (arg == "--query" && retrieval_check && i + 1 < argc) {
            queries.emplace_back(argv[++i]);
        } else if (arg == "--db" && (retrieval_check || sticker_tool || feedback_tool) && i + 1 < argc) {
            database_path = argv[++i];
        } else if (arg == "--labels" && retrieval_check && i + 1 < argc) {
            labels_path = argv[++i];
        } else if (arg == "--backfill" && feedback_tool) {
            feedback_options.backfill = true;
        } else if (arg == "--days" && feedback_tool && i + 1 < argc) {
            feedback_options.days = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--eval" && feedback_tool) {
            feedback_options.eval = true;
        } else if (arg == "--report" && feedback_tool) {
            feedback_options.report = true;
        } else if (arg == "--import-manual" && feedback_tool && i + 1 < argc) {
            feedback_options.import_manual = argv[++i];
        } else if (arg == "--retag" && sticker_tool) {
            stickers.retag = true;
        } else if (arg == "--gifs" && sticker_tool) {
            stickers.gifs = true;
        } else if (arg == "--prompt" && sticker_tool && i + 1 < argc) {
            stickers.prompt = argv[++i];
        } else if (arg == "--learn-usage" && sticker_tool) {
            stickers.learn_usage = true;
        } else if (arg == "--forget-group" && sticker_tool && i + 1 < argc) {
            stickers.forget_group = std::atoll(argv[++i]);
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
    if (replay && !replay_log && !probe_db) {
        print_usage(stderr);
        return 2;
    }

    const auto path = config_path.value_or(default_config_path());
    // Listing learned expressions reads the database only; no keys needed.
    auto config = load_config(path, !(list_expressions || jargon || (sticker_tool && !stickers.retag && !stickers.learn_usage) ||
                                      (feedback_tool && !feedback_options.backfill)) ||
                                        review);
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

    // Only a plain `lilia [--config]` runs the bot: a subcommand must never
    // fall through to it (a tool run with the live config would join the
    // groups and migrate the live database).
    const bool subcommand = argc > 1 && !std::string_view(argv[1]).starts_with('-');
    const bool tool = replay || learn_log || learn_history || review || retrieval_check || sticker_tool || feedback_tool;
    if (subcommand && !tool) {
        log::error("unhandled subcommand {}", argv[1]);
        return 2;
    }

    lighter::EventLoop loop;
    auto interrupts = lighter::InterruptSource::create(loop);
    if (!interrupts) {
        log::error("cannot watch process signals: {}", interrupts.error().message());
        return 1;
    }
    auto app = lighter::with_token(
        probe_db            ? run_db_probes(*config, *probe_db, persona_path, probe_repeat, probe_limit, planner_prompt) :
        replay_log && probe ? run_probes(*config, *replay_log, persona_path, probe_repeat, planner_prompt) :
        replay_log      ? run_replay(*config, *replay_log, persona_path, habits_db, writer_prompt, stickers_db, continue_at, !free_turn) :
        learn_log       ? run_learn(*config, *learn_log) :
        learn_history   ? run_learn_history(*config, history_groups, history_count) :
        review          ? review_expressions(*config, apply, keep) :
        retrieval_check ? run_retrieval(*config, database_path.value_or(database), queries, labels_path) :
        sticker_tool    ? run_stickers(*config, database_path, stickers) :
        feedback_tool   ? run_feedback(*config, database_path, database, feedback_options) :
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
