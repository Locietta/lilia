#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include <lighter/async/runtime/task.h>

#include <lilia/config.h>
#include <lilia/onebot/types.h>
#include <lilia/provider/history.h>
#include <lilia/provider/provider.h>
#include <lilia/store/store.h>

namespace lilia::learn {

// Expression learning, after MaiBot: patterns "当[情境]时，可以[说法]"
// distilled from group chat. Every group the account is in may feed it, but
// only as a reader: the buffered text lives in memory and is dropped after
// each batch; only the short patterns are stored.

struct Line {
    i64 sender = 0;
    std::string text;
};

struct Learned {
    std::string situation;
    std::string style;
};

struct LearnerRequest {
    provider::History history;
    std::vector<provider::ToolDefinition> tools;
};

/// Speakers become 群友1, 群友2 ... in order of appearance; lines are numbered.
LearnerRequest build_learner_request(std::string_view prompt, const std::vector<Line> &lines);

/// Reads record_expressions; drops items that are empty, too long, or cite a
/// line that does not exist.
std::vector<Learned> parse_learned(const std::vector<provider::OutputItem> &items, usize line_count);

/// Text worth learning from: plain text segments only (no media, mentions or
/// replies), trimmed; empty when the message has none.
std::string learnable_text(const onebot::Message &message);

struct Learner {
    using Spawn = std::move_only_function<void(lighter::Task<>)>;
    using Clock = std::move_only_function<i64()>;

    Learner(const LearningConfig &config, std::string name, store::Store &store, provider::Provider provider, Spawn spawn);

    /// Buffers a group message (lilia's own excluded); every `batch_messages`
    /// lines of one group start a learning call.
    void observe(const onebot::MessageEvent &message, i64 self_id);

    /// One extraction call over `lines`; stores what it learns. Returns it.
    lighter::Task<std::vector<Learned>> learn(i64 group, std::vector<Line> lines);

    Clock now;
    const LearningConfig &config;
    std::string prompt;
    store::Store &store;
    provider::Provider provider;
    Spawn spawn;
    std::map<i64, std::vector<Line>> buffers;
};

} // namespace lilia::learn
