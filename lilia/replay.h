#pragma once

#include <filesystem>
#include <optional>

#include <lighter/async/runtime/task.h>

#include <lilia/config.h>
#include <lilia/provider/provider.h>

namespace lilia {

/// Anthropic-compatible clients. The planner must call a tool; the searcher
/// uses the planner's model with the hosted web_search tool.
provider::Provider make_planner(const ModelConfig &model);
provider::Provider make_searcher(const ModelConfig &model);
provider::Provider make_writer(const ModelConfig &model);
/// Expression learner: the planner's model, must call record_expressions.
provider::Provider make_learner(const ModelConfig &model);

/// Feeds a JSONL chat log through the real gate, planner and writer (no
/// NapCat, in-memory store, no delays) and prints every decision. Each line:
///   {"t": 12, "uid": 2001, "name": "小明", "text": "@莉莉娅 在吗"}
/// `t` is seconds from the start. "@name" at the start of `text`, where name
/// is lilia's name, an alias or `nickname`, becomes a mention.
/// `habits`: a lilia.db (opened read-only) whose learned expressions are
/// copied into the replay's in-memory store.
lighter::Task<int> run_replay(Config config, std::filesystem::path log, std::optional<std::filesystem::path> persona,
                              std::optional<std::filesystem::path> habits);

/// Runs the expression learner over a JSONL log (same format as replay) in
/// batches of `learning.batch_messages` and prints what it would learn.
lighter::Task<int> run_learn(Config config, std::filesystem::path log);

/// Prints the learned expressions of a lilia.db (read-only).
int print_expressions(const std::filesystem::path &database, usize limit);

} // namespace lilia
