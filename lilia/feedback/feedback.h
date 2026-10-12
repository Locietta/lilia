#pragma once

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/async/runtime/task.h>

#include <lilia/chat/addressee.h>
#include <lilia/config.h>
#include <lilia/feedback/logistic.h>
#include <lilia/learn/learner.h>
#include <lilia/provider/provider.h>
#include <lilia/store/store.h>

namespace lilia::feedback {

// Hindsight feedback, after MaiBot's reply_effect: a while after she spoke
// (or once enough lines have piled up), the helper model reads the window
// with what came after it and says whom each line was said to, and how her
// messages went down. The addressee labels train a per-group prior
// (logistic.h) that the planner sees as a hint, never a gate; the reactions
// to her stickers weight them in search. Only chats she answers in, whose
// lines are stored anyway; labels keep features, not text.

struct LineVerdict {
    i64 row = 0;
    std::string to; ///< "her", "other", "group", "self" or "unclear"
};

struct ReplyVerdict {
    i64 row = 0;
    std::string for_her;           ///< "yes", "no", "unclear"
    std::string answered_by_other; ///< "yes", "no", "unclear"
    std::string stance;
    std::string contribution;
};

struct Hindsight {
    std::vector<LineVerdict> lines;
    std::vector<ReplyVerdict> replies;
};

/// The judge call (tool record_hindsight): the window as the planner sees it,
/// the rows to label, and her messages with the row each answered.
learn::LearnerRequest build_hindsight_request(std::string_view prompt, std::string_view lines, std::string_view targets,
                                              std::string_view own);

/// nullopt when the tool was not called; unknown values become "unclear".
std::optional<Hindsight> parse_hindsight(const std::vector<provider::OutputItem> &items);

/// How a message of hers went down: reactions, not how long the chat ran
/// (Pang et al. 2023). appreciation +1 … attack -1, plus advance +0.5 …
/// intrusion -1.
double reward_of(std::string_view stance, std::string_view contribution);

/// "低" below 0.2, "中" up to 0.6, "高" above.
std::string_view level(double probability);

struct Feedback {
    using Spawn = std::move_only_function<void(lighter::Task<>)>;

    Feedback(const FeedbackConfig &config, const ChatConfig &chat, store::Store &store, provider::Provider judge, Spawn spawn);

    /// The fitted models, from the store.
    void load();

    /// After a line was stored in an allowed chat: judges the chat's window
    /// once it is due (rate- and budget-limited).
    void observe(const store::ChatKey &chat, i64 self_id);

    /// Judges the next window of `chat` (up to `limit` lines); false when
    /// nothing was judged (the call failed or there was nothing to judge).
    lighter::Task<bool> judge(store::ChatKey chat, i64 self_id, usize limit);

    /// Fits the global and per-group models from the labels and stores them.
    void refit();

    /// The model for `chat`: its own, else the global one, else bootstrap.
    const Weights &model_for(const store::ChatKey &chat) const;
    bool fitted(const store::ChatKey &chat) const;

    struct Hint {
        std::string shown;  ///< for the planner's tail; empty when hints are off
        std::string logged; ///< "#12 0.23, #13 0.71" for planner_turns
    };
    /// The prior for new lines (from `first_new_row`) that point at no one
    /// explicitly; edges align with rows.
    Hint hints(const store::ChatKey &chat, const std::vector<store::StoredMessage> &rows, const std::vector<chat::LineEdges> &edges,
               i64 first_new_row) const;

    std::move_only_function<i64()> now;
    const FeedbackConfig &config;
    const ChatConfig &chat;
    store::Store &store;
    provider::Provider judge_model;
    std::string judge_model_name;
    Spawn spawn;
    std::string prompt;
    /// learning.daily_token_budget (0: none); judging is background work.
    u64 background_budget = 0;
    std::map<store::ChatKey, Weights> models;
    Weights global = bootstrap_weights();
    bool global_fitted = false;
    std::set<store::ChatKey> judging;
    i64 hour = 0;
    usize judges_this_hour = 0;
    usize labels_since_refit = 0;
};

} // namespace lilia::feedback
