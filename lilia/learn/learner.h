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
#include <lilia/retrieval/retriever.h>
#include <lilia/store/store.h>

namespace lilia::learn {

// Learning from group chat, after MaiBot's expression and jargon learners.
// Every group the account is in feeds per-group batches (lilia's own lines
// ride along as [SELF] context and are never learned from). Each batch gives
// two calls:
// - expressions: "当[情境]时，可以[说法]" patterns, filtered, reviewed with
//   reasons (all verdicts logged; rejected ones can be rescued by hand),
//   stored per source chat with merged situations and an embedding;
// - jargon: terms that need in-group context, plus sightings of known terms;
//   each sighting keeps an anonymised snippet as evidence, and at 4, 8, 25
//   and 100 sightings the meaning is inferred twice (from context, from the
//   word alone) and compared: the same meaning means an ordinary word.
// Batches of the chats she talks in also feed their high-frequency terms
// (frequency.h), which rank slang in the jargon reference.

struct Line {
    i64 sender = 0;
    std::string text;
    bool self = false; ///< lilia's own line, context only
    i64 time = 0;      ///< unix seconds; 0 when unknown
};

struct Learned {
    std::string situation;
    std::string style;
    usize source = 0; ///< 0-based line index
};

struct Rejected {
    usize index = 0; ///< into the reviewed list
    std::string reason;
};

struct Candidate {
    std::string content;
    usize source = 0; ///< 0-based line index
};

struct Meaning {
    std::string meaning;
    bool no_info = false;
};

struct LearnerRequest {
    provider::History history;
    std::vector<provider::ToolDefinition> tools;
};

/// The input of the first call of `tool` among the answer's items; its
/// fields are read with lilia/json.h.
const glz::generic *tool_input(const std::vector<provider::OutputItem> &items, std::string_view tool);

/// Speakers become 群友1, 群友2 ... in order of appearance, lilia's lines
/// "[SELF] 名字"; lines are numbered from 1.
std::string render_lines(const std::vector<Line> &lines, std::string_view self_name);

LearnerRequest build_learner_request(std::string_view prompt, const std::vector<Line> &lines, std::string_view self_name);

/// Reads record_expressions. nullopt: more than 20 items, the whole batch is
/// discarded (MaiBot). Drops items citing a missing or [SELF] line, mentioning
/// SELF/images/stickers, equal to one of `names`, too long, or put-downs, and
/// styles whose quoted words (each part of 用"…", cut at ellipses) are not
/// in their source line; those last are counted in `ungrounded`.
std::optional<std::vector<Learned>> parse_learned(const std::vector<provider::OutputItem> &items, const std::vector<Line> &lines,
                                                  const std::vector<std::string> &names, usize *ungrounded = nullptr);

/// True for patterns that put people down, which lilia must never pick up:
/// outright slurs always; words like 区/菜/垃圾 only when aimed at a person
/// (a situation about 别人/对方/某人, or a phrase said to 你), since they are
/// fine about cards, items and other things. ASCII words (sb) match whole,
/// single characters only as words of their own (区, not 区别).
bool blocked_expression(std::string_view situation, std::string_view style);

/// One spelling per style: 「」“”『』 become ASCII quotes and 说"…"/回"…"/使用"…"
/// become 用"…", so the same phrase learned twice counts up.
std::string normalize_style(std::string_view style);

/// The review (MaiBot's suitability check): which of `learned` to drop and
/// why (tool review_expressions, field reject: [{id, reason}]).
LearnerRequest build_review_request(std::string_view prompt, const std::vector<Learned> &learned);

/// nullopt when the review tool was not called.
std::optional<std::vector<Rejected>> parse_review(const std::vector<provider::OutputItem> &items, usize count);

LearnerRequest build_jargon_request(std::string_view prompt, const std::vector<Line> &lines, std::string_view self_name);

/// Words in the jieba dictionary with frequency >= 500 are common everyday words.
inline constexpr i64 k_max_jargon_dict_frequency = 500;

/// Returns true if term is all ASCII digits/punctuation/spaces (e.g. "200"),
/// or if dictionary_frequency(term) >= k_max_jargon_dict_frequency (checked
/// for both term and normalize_match_text(term)).
bool ordinary_jargon_word(std::string_view term);

/// Reads record_jargon: 2-12 characters, a real line that contains the term,
/// not a name; at most 30, deduplicated.
std::vector<Candidate> parse_jargon(const std::vector<provider::OutputItem> &items, const std::vector<Line> &lines,
                                    const std::vector<std::string> &names);

/// Known terms (2+ characters) that occur in others' lines, with the first
/// line each occurs in; ASCII terms must stand as whole words.
std::vector<Candidate> known_hits(const std::vector<Line> &lines, const std::vector<std::string> &terms);

/// The line and three on each side, speakers anonymised as in render_lines.
std::string evidence_snippet(const std::vector<Line> &lines, usize source, std::string_view self_name);

/// Inference is due when a threshold (4, 8, 25, 100) lies in
/// (last_inference_count, count].
bool inference_due(i64 count, i64 last_inference_count);

LearnerRequest build_inference_request(std::string_view prompt, const store::Jargon &term,
                                       const std::vector<store::JargonEvidence> &evidence, std::string_view self_name);
LearnerRequest build_word_inference_request(std::string_view prompt, std::string_view content);
std::optional<Meaning> parse_meaning(const std::vector<provider::OutputItem> &items);
LearnerRequest build_compare_request(std::string_view prompt, std::string_view first, std::string_view second);
/// nullopt when record_comparison was not called.
std::optional<bool> parse_similar(const std::vector<provider::OutputItem> &items);

/// Whether chats `a` and `b` share learned material under `groups` (each a
/// sharing group; 0 means every chat). Empty `groups`: everything is shared;
/// `b` == 0 (learned before chats were recorded) is shared everywhere.
bool shares(const std::vector<std::vector<i64>> &groups, i64 a, i64 b);

/// What a pattern is embedded as (MaiBot's form): "情景：…\n风格：…".
std::string expression_text(std::string_view situation, std::string_view style);

/// Every pattern as embedded, for the startup backfill.
std::vector<std::string> embedded_texts(store::Store &store);

/// "以下黑话来自…" lines for the slang (is_jargon, with a meaning) that occurs
/// in others' messages of `log`, at most `limit`; empty when none occurs.
/// Ranked as MaiBot does: by count, but slang that is also one of the chat's
/// `frequent` terms comes first (marked "同时命中高频词"); ties go to the
/// term seen earlier, then the longer one.
std::string jargon_reference(const std::vector<store::Jargon> &terms, const std::vector<store::StoredMessage> &log,
                             const std::vector<store::FrequentTerm> &frequent, usize limit);

/// What the selector sees at reply time: recent chat lines ("昵称: 内容"),
/// the message she answers, her intent, and candidates "当X时，可以Y".
struct SelectionInput {
    std::vector<std::string> recent;
    std::string target;
    std::string intent;
    std::vector<std::string> candidates;
    usize max_picks = 3;
};

/// The selector call: picks 0 to `max_picks` candidates that fit this reply.
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

    Learner(const LearningConfig &config, std::string name, std::vector<std::string> aliases, store::Store &store,
            provider::Provider provider, Spawn spawn);

    /// Buffers a group message (her own excluded; see observe_own); enough
    /// lines of one group start a batch, rate- and concurrency-limited.
    void observe(const onebot::MessageEvent &message, i64 self_id);

    /// Her own line in a group, as [SELF] context for the next batch.
    void observe_own(i64 group, std::string text);

    /// One batch: frequent terms (chats in `frequency_chats` only),
    /// expressions (extract, review, store) and slang. Returns the stored
    /// patterns. The lines' times widen the group's learned span.
    lighter::Task<std::vector<Learned>> learn(i64 group, std::vector<Line> lines);

    /// Others' lines of one batch into `chat`'s high-frequency terms.
    void count_frequent_terms(const store::ChatKey &chat, const std::vector<Line> &lines);

    /// The review call alone; nullopt when it failed. Recorded for `group`.
    lighter::Task<std::optional<std::vector<Rejected>>> review(i64 group, const std::vector<Learned> &learned);

    /// Slang from one batch: candidates, known terms, evidence, inference.
    lighter::Task<> learn_jargon(i64 group, std::vector<Line> lines);

    /// Infers `term`'s meaning (context, word alone, comparison) and stores it.
    lighter::Task<> infer(store::Jargon term);

    Clock now;
    const LearningConfig &config;
    std::string name;
    std::vector<std::string> names; ///< her name and aliases
    std::string prompt;
    std::string review_prompt;
    std::string jargon_prompt;
    std::string infer_prompt;
    std::string word_prompt;
    std::string compare_prompt;
    store::Store &store;
    provider::Provider provider;
    Spawn spawn;
    /// Embeds new patterns when set (memory on).
    retrieval::Retriever *retriever = nullptr;
    /// Chats whose high-frequency terms are kept: the ones she talks in, the
    /// only place they are used (the jargon reference).
    std::vector<i64> frequency_chats;
    std::map<i64, std::vector<Line>> buffers;
    std::map<i64, i64> last_batch;
    usize running = 0;
};

} // namespace lilia::learn
