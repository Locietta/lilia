#pragma once

#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <lighter/async/runtime/task.h>

#include <lilia/archive/archiver.h>
#include <lilia/chat/context.h>
#include <lilia/chat/flow.h>
#include <lilia/config.h>
#include <lilia/learn/learner.h>
#include <lilia/memory/memory.h>
#include <lilia/sticker/sticker.h>
#include <lilia/onebot/session.h>
#include <lilia/provider/provider.h>
#include <lilia/store/store.h>

namespace lilia::chat {

/// Turns OneBot events into stored messages and replies. Lives for the whole
/// process; sessions come and go as NapCat reconnects.
///
/// A turn: the flow gate decides that the planner should look; after the
/// burst settles, the planner works in rounds, as MaiBot's does: each round
/// it replies, waits, looks something up or ends the turn. After a reply or
/// a wait it sees what she said and the messages that came in meanwhile. A
/// reply's text comes from the writer and goes out as 1-3 messages.
struct Responder {
    using Spawn = std::move_only_function<void(lighter::Task<>)>;
    using Clock = std::move_only_function<i64()>;
    /// Sends one message; returns its id, or nullopt when sending failed.
    using Outbox = std::move_only_function<lighter::Task<std::optional<i64>>(const store::ChatKey &, onebot::Message)>;
    /// Observes every turn decision (replay output, tests).
    using PlanObserver = std::move_only_function<void(const store::ChatKey &, const Plan &, const std::vector<std::string> &)>;
    /// Lets `seconds` pass during a planner wait.
    using Pause = std::move_only_function<lighter::Task<>(const store::ChatKey &, i64 seconds)>;

    Responder(const Config &config, store::Store &store, provider::Provider planner, provider::Provider searcher, provider::Provider writer,
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

    /// One turn of planner rounds and the replies they ask for. Consumes the
    /// chat's unseen messages, including those arriving during the turn.
    lighter::Task<> run_turn(store::ChatKey chat);

    Result<void> reload_persona();

    FlowSettings flow_settings() const;

    /// Unix seconds; replaceable for tests and replay.
    Clock now;
    /// Replaces the live session for sending (replay).
    Outbox outbox;
    PlanObserver observer;
    /// How a planner wait passes; unset, it sleeps (when realtime). Replay
    /// advances its clock and feeds the messages of that time.
    Pause pause;
    /// Debounce and typing delays; replay turns them off.
    bool realtime = true;
    /// Blind comparisons (replay --continue-at): until she has replied, the
    /// planner may only reply (or look something up).
    bool must_speak = false;

    struct ChatState {
        Flow flow;
        std::string name;
        bool busy = false;
        i64 budget_notice_day = -1;
        /// Her latest planner decisions, oldest first, shown to the planner.
        std::deque<std::string> decisions;
        /// What she said to recent targets (message id, her text), newest last.
        std::deque<std::pair<i64, std::string>> replied;
        /// When she last sent a sticker here (unix seconds).
        i64 last_sticker = 0;
    };

    const Config &config;
    store::Store &store;
    provider::Provider planner;
    provider::Provider searcher;
    provider::Provider writer;
    /// Tried in order when `writer` fails or returns nothing.
    struct NamedProvider {
        std::string model;
        provider::Provider provider;
    };
    std::vector<NamedProvider> writer_fallbacks;
    std::filesystem::path persona_path;
    std::string persona;
    /// The writer's rules (writer.md); replays may swap in a variant.
    std::string writer_rules;
    Spawn spawn;
    /// Archives media of allowed chats; null when archiving is disabled.
    archive::Archiver *archiver = nullptr;
    /// Searchable history of the allowed chats; null when disabled.
    memory::Memory *memory = nullptr;
    /// Collects stickers from every group and holds the pool she sends from;
    /// null when disabled.
    sticker::Collector *stickers = nullptr;
    /// Learns expressions from every group (read-only); null when disabled.
    learn::Learner *learner = nullptr;
    std::mt19937 rng{std::random_device{}()};

    std::shared_ptr<onebot::Session> session;
    onebot::LoginInfo self;
    std::map<store::ChatKey, ChatState> chats;
};

/// Built-in persona used until the data directory has a persona.md.
std::string_view default_persona();

/// Learned patterns offered to the planner: about a third the most counted,
/// the rest random, shuffled; rendered "当X时，可以Y".
/// MaiBot's expression sample without embeddings: five weighted by count
/// (mapped onto 1..5) from `pool`, five more from patterns learned more than
/// once (when there are enough), without repeats.
std::vector<store::Expression> sample_fallback(std::vector<store::Expression> pool, std::mt19937 &rng);

/// Up to `count` distinct quoted phrases from random learned patterns.
std::vector<std::string> sample_voice(store::Store &store, usize count);

} // namespace lilia::chat
