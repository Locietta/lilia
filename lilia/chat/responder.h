#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <lighter/async/runtime/task.h>

#include <lilia/archive/archiver.h>
#include <lilia/chat/context.h>
#include <lilia/chat/flow.h>
#include <lilia/config.h>
#include <lilia/onebot/session.h>
#include <lilia/provider/provider.h>
#include <lilia/store/store.h>

namespace lilia::chat {

/// Turns OneBot events into stored messages and replies. Lives for the whole
/// process; sessions come and go as NapCat reconnects.
///
/// A turn: the flow gate decides that the planner should look; after the
/// burst settles, the planner (one tool call) decides whether lilia speaks
/// and how; the writer produces the text, which goes out as 1-3 messages.
struct Responder {
    using Spawn = std::move_only_function<void(lighter::Task<>)>;
    using Clock = std::move_only_function<i64()>;
    /// Sends one message; returns its id, or nullopt when sending failed.
    using Outbox = std::move_only_function<lighter::Task<std::optional<i64>>(const store::ChatKey &, onebot::Message)>;
    /// Observes every turn decision (replay output, tests).
    using PlanObserver = std::move_only_function<void(const store::ChatKey &, const Plan &, const std::vector<std::string> &)>;

    Responder(const Config &config, store::Store &store, provider::Provider planner, provider::Provider writer,
              std::filesystem::path persona_path, Spawn spawn);

    void attach(std::shared_ptr<onebot::Session> session, onebot::LoginInfo login);
    void detach() noexcept;

    /// Called on the session's reader loop for every event; never blocks.
    /// Turns run as spawned tasks, at most one per chat at a time.
    void on_event(onebot::Event event);

    /// Stores an allowed message and updates its chat's flow. Returns the
    /// chat when the planner should look at it now. Spawns admin commands
    /// and media archiving.
    std::optional<store::ChatKey> ingest(const onebot::MessageEvent &message);

    /// One planner decision and, if she speaks, one reply. Consumes the
    /// chat's unseen messages.
    lighter::Task<> run_turn(store::ChatKey chat);

    Result<void> reload_persona();

    FlowSettings flow_settings() const;

    /// Unix seconds; replaceable for tests and replay.
    Clock now;
    /// Replaces the live session for sending (replay).
    Outbox outbox;
    PlanObserver observer;
    /// Debounce and typing delays; replay turns them off.
    bool realtime = true;

    struct ChatState {
        Flow flow;
        std::string name;
        bool busy = false;
        i64 budget_notice_day = -1;
    };

    const Config &config;
    store::Store &store;
    provider::Provider planner;
    provider::Provider writer;
    std::filesystem::path persona_path;
    std::string persona;
    Spawn spawn;
    /// Archives media of allowed chats; null when archiving is disabled.
    archive::Archiver *archiver = nullptr;
    std::mt19937 rng{std::random_device{}()};

    std::shared_ptr<onebot::Session> session;
    onebot::LoginInfo self;
    std::map<store::ChatKey, ChatState> chats;
};

/// Built-in persona used until the data directory has a persona.md.
std::string_view default_persona();

} // namespace lilia::chat
