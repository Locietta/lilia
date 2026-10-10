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
/// Image and sticker descriptions (`vision_model` on `base_url`).
provider::Provider make_vision(const ModelConfig &model);
/// Expression learner: the planner's model, must call record_expressions.
provider::Provider make_learner(const ModelConfig &model);

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
/// after the log up to it; she must say something (the planner still picks
/// whom and what). Printed last as "RESULT {json}" (replied, messages,
/// decisions), for blind comparisons with what people really said.
lighter::Task<int> run_replay(Config config, std::filesystem::path log, std::optional<std::filesystem::path> persona,
                              std::optional<std::filesystem::path> habits, std::optional<std::filesystem::path> writer_prompt = {},
                              std::optional<std::filesystem::path> stickers = {}, std::optional<usize> continue_at = {});

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
