#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
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

/// True for patterns that put people down, which lilia must never pick up:
/// outright slurs always; words like 区/菜/垃圾 only when aimed at a person
/// (a situation about 别人/对方/某人, or a phrase said to 你), since they are
/// fine about cards, items and other things.
bool blocked_expression(std::string_view situation, std::string_view style);

/// One spelling per style: 「」“”『』 become ASCII quotes and 说"…"/回"…"
/// become 用"…", so the same phrase learned twice counts up.
std::string normalize_style(std::string_view style);

/// The review step before storing (MaiBot's suitability check): which of
/// `learned` to drop (tool review_expressions, field reject). Numbered from 1.
LearnerRequest build_review_request(std::string_view prompt, const std::vector<Learned> &learned);

/// What the selector sees at reply time: recent chat lines ("昵称: 内容"),
/// the message she answers, her intent, and candidates "当X时，可以Y".
struct SelectionInput {
    std::vector<std::string> recent;
    std::string target;
    std::string intent;
    std::vector<std::string> candidates;
};

/// The selector call: picks 0 to 3 candidates that fit this reply.
LearnerRequest build_selection_request(std::string_view prompt, const SelectionInput &input);

/// Reads the number list `field` of tool `tool` ("3", 3 or "#3"), as 0-based
/// indices below `count`, deduplicated, at most `limit`. Empty when the tool
/// was not called.
std::vector<usize> parse_picks(const std::vector<provider::OutputItem> &items, std::string_view tool, std::string_view field, usize count,
                               usize limit);

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

    /// One extraction call over `lines`, then a review call; stores what
    /// passes the review and returns it.
    lighter::Task<std::vector<Learned>> learn(i64 group, std::vector<Line> lines);

    /// The review call alone: which of `learned` to keep (indices), nullopt
    /// when the call failed. Recorded as a learning call for `group`.
    lighter::Task<std::optional<std::vector<usize>>> review(i64 group, const std::vector<Learned> &learned);

    Clock now;
    const LearningConfig &config;
    std::string prompt;
    std::string review_prompt;
    store::Store &store;
    provider::Provider provider;
    Spawn spawn;
    std::map<i64, std::vector<Line>> buffers;
};

} // namespace lilia::learn
