#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <lighter/types.hpp>

#include <lilia/error.h>

namespace lilia {

using namespace lighter::types;

struct OneBotConfig {
    std::string host = "127.0.0.1"; ///< numeric address of NapCat's websocket server
    i32 port = 3001;
    std::string path = "/";
    std::string access_token; ///< secret; usually "$LILIA_ONEBOT_TOKEN"
};

struct ModelConfig {
    std::string base_url = "https://api.deepseek.com/anthropic";
    std::string model = "deepseek-flash";
    std::string api_key; ///< secret; usually "$DEEPSEEK_API_KEY"
    u32 max_tokens = 2048;
    std::optional<std::string> reasoning_effort;
};

struct ChatConfig {
    /// Only these groups are logged and answered.
    std::vector<i64> groups;
    /// Private chats are answered only for these QQ numbers.
    std::vector<i64> private_users;
    /// May use /lilia admin commands.
    std::vector<i64> admins;
    /// Calling the bot by any of these names (case-insensitive) triggers a reply.
    std::vector<std::string> aliases = {"lilia", "莉莉娅"};
    /// Recent messages rendered into the model context.
    usize context_messages = 40;
    /// Byte budget for the rendered chat log; older lines are dropped first.
    usize context_bytes = 12'000;
    u32 min_reply_interval_seconds = 3;
    /// Input plus output tokens per local calendar day across all chats.
    u64 daily_token_budget = 2'000'000;
    usize max_reply_bytes = 1'500;
};

struct Config {
    OneBotConfig onebot;
    ModelConfig model;
    ChatConfig chat;
    /// Database and persona location; empty means the platform default.
    std::string data_dir;
    std::string log_level = "info";
};

/// Reads JSON (comments allowed). Unknown keys are errors so typos surface.
/// Secret fields of the form "$NAME" are replaced from the environment; a
/// literal secret is only accepted when the file is private to its owner.
Result<Config> load_config(const std::filesystem::path &path);

/// Parses config text without touching the file system or environment.
Result<Config> parse_config(std::string_view text);

/// "$NAME" -> value of NAME (error when unset or empty); anything else as-is.
Result<std::string> resolve_secret(std::string_view value, std::string_view field);

/// $XDG_DATA_HOME/lilia or ~/.local/share/lilia; %LOCALAPPDATA%/lilia on Windows.
std::filesystem::path default_data_dir();

/// ~/.config/lilia/config.json (XDG); %APPDATA%/lilia/config.json on Windows.
std::filesystem::path default_config_path();

} // namespace lilia
