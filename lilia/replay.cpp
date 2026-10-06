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
#include <lilia/onebot/message.h>
#include <lilia/provider/anthropic.h>
#include <lilia/store/store.h>

namespace lilia {

using lighter::Task;

namespace {

constexpr i64 k_replay_group = 1;
constexpr i64 k_replay_self = 10'001;

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

provider::Provider make_provider(const ModelConfig &model, const std::string &model_name, u32 max_tokens) {
    return pro::make_proxy<provider::ProviderFacade, anthropic::Client>(anthropic::ClientOptions{
        .auth = provider::api_key_auth(model.api_key, false),
        .base_url = model.base_url,
        .model = model_name,
        .reasoning_effort = model.reasoning_effort,
        .disable_thinking = !model.thinking,
        .max_tokens = max_tokens,
    });
}

Task<int> run_replay(Config config, std::filesystem::path log, std::optional<std::filesystem::path> persona) {
    std::ifstream input(log, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot read %s\n", log.string().c_str());
        co_return 1;
    }
    std::vector<ReplayLine> lines;
    std::string raw;
    for (usize number = 1; std::getline(input, raw); ++number) {
        if (raw.find_first_not_of(" \t\r") == std::string::npos) continue;
        ReplayLine line;
        if (auto ec = glz::read<glz::opts{.null_terminated = false, .error_on_unknown_keys = false}>(line, raw)) {
            std::fprintf(stderr, "%s:%zu: %s\n", log.string().c_str(), number, glz::format_error(ec, raw).c_str());
            co_return 1;
        }
        lines.push_back(std::move(line));
    }

    config.chat.groups = {k_replay_group};
    config.chat.private_users.clear();
    config.chat.admins.clear();
    auto store = store::Store::open_memory();
    if (store.has_error()) {
        std::fprintf(stderr, "%s\n", store.error().message().c_str());
        co_return 1;
    }

    lighter::TaskGroup tasks(lighter::EventLoop::current());
    chat::Responder responder(config, *store, make_provider(config.model, config.model.planner_model, config.model.planner_max_tokens),
                              make_provider(config.model, config.model.model, config.model.max_tokens),
                              persona.value_or(std::filesystem::path(config.data_dir) / "persona.md"),
                              [&tasks](Task<> task) { tasks.spawn(std::move(task)); });
    if (auto loaded = responder.reload_persona(); loaded.has_error()) {
        std::fprintf(stderr, "%s\n", loaded.error().message().c_str());
        co_return 1;
    }
    responder.self = {.user_id = k_replay_self, .nickname = config.chat.name};
    responder.realtime = false;

    const i64 base = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    i64 sim = 0;
    responder.now = [&sim, base] { return base + sim; };
    i64 next_id = 1'000'000;
    responder.outbox = [&next_id, &sim](const store::ChatKey &, onebot::Message message) -> Task<std::optional<i64>> {
        std::string quote;
        if (onebot::reply_target(message)) quote = "↩ ";
        print(std::format("[{}]   {}: {}{}", clock(sim), "莉莉娅", quote, onebot::render_plain(message)));
        co_return ++next_id;
    };
    responder.observer = [](const store::ChatKey &, const chat::Plan &plan, const std::vector<std::string> &) {
        if (!plan.reply) {
            print(std::format("        · 不说话：{}", plan.reason));
        } else {
            print(std::format("        · 回复 {}（{}{}{}）：{}", plan.target, style_name(plan.style), plan.quote ? "，引用" : "",
                              plan.search ? "，搜索" : "", plan.intent));
        }
    };

    std::vector<std::string> names = config.chat.aliases;
    names.push_back(config.chat.name);
    i64 message_id = 0;
    for (const auto &line : lines) {
        sim = line.t;
        print(std::format("[{}] {}: {}", clock(sim), line.name, line.text));
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
        if (const auto chat = responder.ingest(event)) co_await responder.run_turn(*chat);
    }
    co_await tasks.join();
    co_return 0;
}

} // namespace lilia
