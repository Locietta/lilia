#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <lighter/async/runtime/task.h>

#include <lilia/config.h>
#include <lilia/onebot/session.h>
#include <lilia/provider/provider.h>
#include <lilia/store/store.h>

namespace lilia::chat {

/// Turns OneBot events into stored messages and model replies. Lives for the
/// whole process; sessions come and go as NapCat reconnects.
struct Responder {
    using Spawn = std::move_only_function<void(lighter::Task<>)>;
    using Clock = std::move_only_function<i64()>;

    Responder(const Config &config, store::Store &store, provider::Provider provider, std::filesystem::path persona_path, Spawn spawn);

    void attach(std::shared_ptr<onebot::Session> session, onebot::LoginInfo login);
    void detach() noexcept;

    /// Called on the session's reader loop for every event; never blocks.
    /// Replies run as spawned tasks, at most one per chat at a time.
    void on_event(onebot::Event event);

    Result<void> reload_persona();

    /// Unix seconds; replaceable for tests.
    Clock now;

    struct ChatState {
        std::vector<store::StoredMessage> pending;
        std::string name;
        bool busy = false;
        i64 last_reply = 0;
        i64 budget_notice_day = -1;
    };

    const Config &config;
    store::Store &store;
    provider::Provider provider;
    std::filesystem::path persona_path;
    std::string persona;
    Spawn spawn;

    std::shared_ptr<onebot::Session> session;
    onebot::LoginInfo self;
    std::map<store::ChatKey, ChatState> chats;
};

/// Built-in persona used until the data directory has a persona.md.
extern const std::string_view k_default_persona;

} // namespace lilia::chat
