#pragma once

#include <filesystem>
#include <optional>

#include <lighter/async/runtime/task.h>

#include <lilia/config.h>
#include <lilia/provider/provider.h>

namespace lilia {

/// Anthropic-compatible clients for the planner and the writer.
provider::Provider make_planner(const ModelConfig &model);
provider::Provider make_writer(const ModelConfig &model);

/// Feeds a JSONL chat log through the real gate, planner and writer (no
/// NapCat, in-memory store, no delays) and prints every decision. Each line:
///   {"t": 12, "uid": 2001, "name": "小明", "text": "@莉莉娅 在吗"}
/// `t` is seconds from the start. "@name" at the start of `text`, where name
/// is lilia's name, an alias or `nickname`, becomes a mention.
lighter::Task<int> run_replay(Config config, std::filesystem::path log, std::optional<std::filesystem::path> persona);

} // namespace lilia
