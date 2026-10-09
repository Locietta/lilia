#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <xmake/version/lilia.hpp>

#include <lighter/async/async.h>
#include <lighter/async/io/loop.h>
#include <lighter/async/runtime/interrupt.h>

#include <lilia/archive/archiver.h>
#include <lilia/chat/responder.h>
#include <lilia/config.h>
#include <lilia/learn/frequency.h>
#include <lilia/learn/learner.h>
#include <lilia/learn/segment.h>
#include <lilia/log.h>
#include <lilia/onebot/session.h>
#include <lilia/replay.h>
#include <lilia/store/store.h>

namespace {

using namespace lilia;
using lighter::Task;
using namespace std::chrono_literals;

void print_usage(FILE *output) {
    std::fputs("usage: lilia [--config <path>] [--check-config]\n"
               "       lilia replay <chat.jsonl> [--config <path>] [--persona <persona.md>] [--habits <lilia.db>]\n"
               "                    [--writer-prompt <writer.md>] [--stickers <lilia.db>]\n"
               "       lilia learn <chat.jsonl> [--config <path>]\n"
               "       lilia expressions [--config <path>] [--limit <n>]\n"
               "       lilia expressions --review [--apply] [--keep <text>]... [--config <path>]\n"
               "       lilia expressions --rejected | --rescue <review id> | --check <id> | --uncheck <id>\n"
               "       lilia jargon [--limit <n>] | --set <term> <meaning> | --delete <term>\n"
               "       lilia segment [--terms | --count <n>] < lines.txt\n"
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
    archive::Archiver archiver(config.archive, data_dir, *store);
    if (!config.model.vision_model.empty()) {
        archiver.vision = make_vision(config.model);
        archiver.vision_model = config.model.vision_model;
    }
    if (config.archive.enabled) responder.archiver = &archiver;
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
        responder.stickers = &collector;
    }
    memory::Memory memory(config.memory, config.chat.name, config.chat.utc_offset_minutes, *store);
    if (config.memory.enabled) {
        responder.memory = &memory;
        memory.summarizer = make_learner(config.model);
        memory.summarizer_model = config.model.planner_model;
        // Expressions are embedded for retrieval; catch up on older ones.
        learner.embedder = &memory;
        if (config.learning.enabled) tasks.spawn(learner.embed_missing());
        // Vectors of another model are not comparable; drop them and rebuild.
        if (auto dropped = store->delete_chunks_except(config.memory.embedding_model); dropped.has_error())
            log::error("memory: {}", dropped.error().message());
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
    std::optional<std::filesystem::path> learn_log;
    bool list_expressions = false;
    bool jargon = false;
    Manage manage;
    bool review = false;
    bool apply = false;
    std::vector<std::string> keep;
    usize limit = 100;
    bool check_only = false;
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
            if (mode == "--terms") {
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
    } else if (argc > 1 && std::string_view(argv[1]) == "expressions") {
        list_expressions = true;
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
    auto config = load_config(path, !(list_expressions || jargon) || review);
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
    auto app = lighter::with_token(replay_log ? run_replay(*config, *replay_log, persona_path, habits_db, writer_prompt, stickers_db) :
                                   learn_log  ? run_learn(*config, *learn_log) :
                                   review     ? review_expressions(*config, apply, keep) :
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
