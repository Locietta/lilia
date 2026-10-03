#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <xmake/version/lilia.hpp>

#include <lighter/async/async.h>
#include <lighter/async/io/loop.h>
#include <lighter/async/runtime/interrupt.h>

#include <lilia/chat/responder.h>
#include <lilia/config.h>
#include <lilia/log.h>
#include <lilia/onebot/session.h>
#include <lilia/provider/anthropic.h>
#include <lilia/store/store.h>

namespace {

using namespace lilia;
using lighter::Task;
using namespace std::chrono_literals;

void print_usage(FILE *output) { std::fputs("usage: lilia [--config <path>] [--check-config] | --version | --help\n", output); }

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

    auto provider = pro::make_proxy<provider::ProviderFacade, anthropic::Client>(anthropic::ClientOptions{
        .auth = provider::api_key_auth(config.model.api_key, false),
        .base_url = config.model.base_url,
        .model = config.model.model,
        .reasoning_effort = config.model.reasoning_effort,
        .max_tokens = config.model.max_tokens,
    });

    lighter::TaskGroup tasks(lighter::EventLoop::current());
    chat::Responder responder(config, *store, std::move(provider), data_dir / "persona.md",
                              [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    if (auto loaded = responder.reload_persona(); loaded.has_error()) {
        log::error("{}", loaded.error().message());
        co_return 1;
    }

    log::info("lilia {} starting; data in {}; {} allowed group(s); model {}", xmake::version::version, data_dir.string(),
              config.chat.groups.size(), config.model.model);
    tasks.spawn(supervise(config, responder));
    co_await tasks.join();
    co_return 0;
}

} // namespace

int main(int argc, char **argv) {
    std::optional<std::filesystem::path> config_path;
    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
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
        } else {
            print_usage(stderr);
            return 2;
        }
    }

    const auto path = config_path.value_or(default_config_path());
    auto config = load_config(path);
    if (config.has_error()) {
        log::error("{}: {}", path.string(), config.error().message());
        return 1;
    }
    log::g_threshold = parse_level(config->log_level);
    if (check_only) {
        log::info("{} is valid", path.string());
        return 0;
    }

    lighter::EventLoop loop;
    auto interrupts = lighter::InterruptSource::create(loop);
    if (!interrupts) {
        log::error("cannot watch process signals: {}", interrupts.error().message());
        return 1;
    }
    auto app = lighter::with_token(run_app(*config), interrupts->token());
    loop.schedule(app);
    loop.run();

    auto result = app.result();
    if (result.is_cancelled()) {
        log::info("stopped");
        return 0;
    }
    return *result;
}
