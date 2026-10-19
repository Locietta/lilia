#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include <lighter/async/runtime/task.h>

#include <lilia/chat/responder.h>
#include <lilia/config.h>
#include <lilia/provider/provider.h>

namespace lilia {

/// Anthropic-compatible clients. The planner must call a tool; the searcher
/// uses the planner's model with the hosted web_search tool.
provider::Provider make_planner(const ModelConfig &model);
provider::Provider make_searcher(const ModelConfig &model);
provider::Provider make_writer(const ModelConfig &model);
std::vector<chat::Responder::NamedProvider> make_writer_fallbacks(const ModelConfig &model);
/// Planner backups: the planner's tool forcing and token limit, each on its
/// own endpoint (empty: `base_url`).
std::vector<chat::Responder::NamedProvider> make_planner_fallbacks(const ModelConfig &model);
/// Expression selection: the planner's model and endpoint, tool forced, no
/// prompt cache (its candidates differ every call).
provider::Provider make_selector(const ModelConfig &model);
/// Keeps the planner's prompt cache warm: the planner with one output token.
provider::Provider make_warmer(const ModelConfig &model);
/// Image and sticker descriptions (`vision_model` on `base_url`).
provider::Provider make_vision(const ModelConfig &model);
/// Expression learner: the planner's model, must call record_expressions.
provider::Provider make_learner(const ModelConfig &model);
/// The hindsight judge: the helper model, tool forced, room for a verdict per line.
provider::Provider make_judge(const ModelConfig &model);

/// Feeds a JSONL chat log through the real gate, planner and writer (no
/// NapCat, in-memory store, no delays) and prints every decision. Each line:
///   {"t": 12, "uid": 2001, "name": "小明", "text": "@莉莉娅 在吗"}
/// `t` is seconds from the start. "@name" at the start of `text`, where name
/// is lilia's name, an alias or `nickname`, becomes a mention.
/// `habits`: a lilia.db (opened read-only) whose learned expressions are
/// copied into the replay's in-memory store. `writer_prompt` replaces the
/// built-in writer.md, for comparing prompt variants. `stickers`: a lilia.db
/// whose registered stickers she may send (read-only; files from its data dir).
/// `continue_at`: only what she would say right before that (0-based) line,
/// after the log up to it (its pictures archived, with memory the chat
/// indexed and what lies above her window summarised, as live). `forced`, she
/// must say something (the planner still picks whom and what); otherwise she
/// may stay silent. Printed last as "RESULT {json}" (replied, messages,
/// decisions), for blind comparisons with what people really said. A log
/// line may carry `segments`, the message as QQ sent it (export-group-chats.py),
/// which lilia then renders herself as live.
lighter::Task<int> run_replay(Config config, std::filesystem::path log, std::optional<std::filesystem::path> persona,
                              std::optional<std::filesystem::path> habits, std::optional<std::filesystem::path> writer_prompt = {},
                              std::optional<std::filesystem::path> stickers = {}, std::optional<usize> continue_at = {},
                              bool forced = true);

/// `lilia replay <log> --probe`: for every line with an `expect`, a fresh chat
/// of its scenario's lines up to it (the last three new) and one turn, then
/// what she did against what was expected; last the tallies: intrusions
/// (spoke where she should not), misses, and how often the planner's to_her
/// agrees with the expected addressee. Lines may quote (`reply_to`), @ (`at`)
/// and be hers (`self`). `repeat`: runs per probe; `planner_prompt` replaces
/// prompts/planner.md, for comparing prompts.
lighter::Task<int> run_probes(Config config, std::filesystem::path log, std::optional<std::filesystem::path> persona, usize repeat,
                              std::optional<std::filesystem::path> planner_prompt = {});

/// `lilia replay --probe-db <copy.db>`: the same probes on real chat, for the
/// lines the hindsight judge (or a person) labelled in a copy of lilia.db
/// (read-only). Each run gets the line's stored context (up to
/// `chat.context_messages` lines before it) in a fresh chat, the last three
/// of others' lines ending at it new, and one turn. Prints, per run, only
/// the chat's last digits, row ids and decisions, never what anyone said
/// (hers included); last the tallies: to_her agreement on the line itself,
/// misses on lines said to her, chip-ins on the others. `limit`: every line
/// said to her and the newest others up to that many in all (0: all). The
/// addressee prior is shown as live (feedback.hint), from its hand-set
/// weights: nothing fitted on these labels.
lighter::Task<int> run_db_probes(Config config, std::filesystem::path database, std::optional<std::filesystem::path> persona, usize repeat,
                                 usize limit, std::optional<std::filesystem::path> planner_prompt = {});

/// The labelled lines `--probe-db` takes from `labels` (in time order): those
/// of groups, every one said to her and the newest others up to `limit` in
/// all (0: all), still in time order.
std::vector<store::AddresseeLabel> pick_probe_labels(std::vector<store::AddresseeLabel> labels, usize limit);

/// The flow of a chat whose stored `rows` (oldest first) end at a probed
/// line: the last three of others' lines (no notices) are unseen, and
/// `addressed` / `named` are set as ingest sets them (`config.groups` must
/// hold the chat). What the lines quote is looked up in `source`.
chat::Flow probe_flow(const std::vector<store::StoredMessage> &rows, i64 self_id, const ChatConfig &config, store::Store &source);

/// Her QQ id, from her stored messages in `groups`; 0 when she never spoke.
i64 self_id_of(store::Store &store, const std::vector<i64> &groups);

/// Runs the expression learner over a JSONL log (same format as replay) in
/// batches of `learning.batch_messages` and prints what it would learn.
lighter::Task<int> run_learn(Config config, std::filesystem::path log);

/// Learns from the groups' recent history, fetched once from NapCat over a
/// second (read-only) connection into the live database: up to `count`
/// newest messages per group (`groups` empty: every group the account is
/// in), skipping excluded groups and the stretch already learned (live or by
/// an earlier run). Prints what was learned.
lighter::Task<int> run_learn_history(Config config, std::vector<i64> groups, usize count);

/// Prints the learned expressions of a lilia.db (read-only).
int print_expressions(const std::filesystem::path &database, usize limit);

/// Hand management of what lilia learned (writes the database).
struct Manage {
    std::string action; ///< expressions: rejected | rescue | check | uncheck; jargon: list | set | delete
    i64 id = 0;
    std::string term;
    std::string meaning;
    usize limit = 100;
};
int manage_expressions(const std::filesystem::path &database, const Manage &request);
int manage_jargon(const std::filesystem::path &database, const Manage &request);

/// Runs the learning review over every stored expression and lists what it
/// would reject (and blocked put-downs); `apply` deletes those rows. Styles
/// containing any `keep` text are never rejected.
lighter::Task<int> review_expressions(Config config, bool apply, std::vector<std::string> keep = {});

} // namespace lilia
