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

/// Another writer to try when the previous one fails or returns nothing.
struct WriterFallback {
    std::string model;
    std::string base_url; ///< empty means ModelConfig::base_url
    std::string api_key;  ///< secret; empty means ModelConfig::api_key
    bool bearer = false;
    std::string thinking_off; ///< empty means "disabled"
};

struct ModelConfig {
    std::string base_url = "https://api.deepseek.com/anthropic";
    std::string model = "deepseek-flash";
    std::string api_key; ///< secret; usually "$DEEPSEEK_API_KEY"
    u32 max_tokens = 2048;
    std::optional<std::string> reasoning_effort;
    /// Decides whether and how lilia speaks; small and fast is enough.
    std::string planner_model = "deepseek-flash";
    u32 planner_max_tokens = 400;
    /// Let the model think before answering (slower; off by default).
    bool thinking = false;
    /// Optional separate endpoint for the writer (`model`), e.g. MiniMax for
    /// style while the planner stays on `base_url`. Empty means the same.
    std::string writer_base_url;
    std::string writer_api_key; ///< secret; e.g. "$MINIMAX_API_KEY"
    /// Send the writer key as "Authorization: Bearer" instead of x-api-key
    /// (some gateways want that).
    bool writer_bearer = false;
    /// What "thinking off" is sent as; empty means "disabled". Newer Claude
    /// models (e.g. claude-sonnet-5-5) reject that and want "between_tools".
    std::string writer_thinking_off;
    /// Optional separate endpoint for the planner (`planner_model`); empty
    /// means `base_url`. Search, vision and learning stay on `base_url`.
    std::string planner_base_url;
    std::string planner_api_key; ///< secret
    bool planner_bearer = false;
    std::string planner_thinking_off; ///< like writer_thinking_off
    /// Force a tool call with tool_choice "any". Some models (e.g.
    /// claude-sonnet-5-5) reject that; without it the prompt still asks for
    /// a tool, and an answer without one counts as staying silent.
    bool planner_force_tool = true;
    /// Deadlines (seconds) for one model call, retries included. Past them a
    /// call counts as failed, so a hung request cannot leave her silent.
    u32 timeout_seconds = 30;
    u32 writer_timeout_seconds = 40;
    u32 search_timeout_seconds = 90;
    /// Tried in order when the writer call fails (outage, rate limit) or
    /// comes back empty.
    std::vector<WriterFallback> writer_fallbacks;
    /// Writer sampling temperature; unset uses the provider default.
    std::optional<double> writer_temperature;
    /// Describes images and stickers (on `base_url`); empty disables it.
    std::string vision_model = "deepseek-flash";
    /// Offer the provider-hosted web_search server tool. A searched answer
    /// costs tens of thousands of input tokens with DeepSeek.
    bool web_search = true;
};

struct ChatConfig {
    /// Only these groups are logged and answered.
    std::vector<i64> groups;
    /// Private chats are answered only for these QQ numbers.
    std::vector<i64> private_users;
    /// May use /lilia admin commands.
    std::vector<i64> admins;
    /// lilia's name in the persona and prompts.
    std::string name = "莉莉娅";
    /// Calling the bot by any of these names (case-insensitive) addresses her.
    std::vector<std::string> aliases = {"lilia", "莉莉娅"};
    /// Recent messages rendered into the model context.
    usize context_messages = 40;
    /// Byte budget for the rendered chat log; older lines are dropped first.
    usize context_bytes = 12'000;
    /// 0..1, how readily she joins a conversation nobody addressed her in.
    /// The planner looks every ceil(1/talk_value) messages; 0 disables.
    double talk_value = 0.3;
    /// After she speaks, every new message is considered for this long.
    i64 conversation_window_seconds = 300;
    /// Wait for a burst of messages to end before deciding.
    u32 debounce_seconds = 3;
    /// Chance that one reply uses a random alternative style (e.g. one word).
    double alt_style_probability = 0.15;
    /// What the planner hands the writer: "intent" (a short stance to voice)
    /// or "reference" (notes the writer may use or ignore, like MaiBot).
    std::string handoff = "intent";
    /// Typing-delay speed for multi-message replies; 0 sends immediately.
    double typing_speed = 1.0;
    /// How long a turn waits for new images to be described.
    u32 image_wait_seconds = 8;
    /// Input plus output tokens per local calendar day across all chats.
    u64 daily_token_budget = 2'000'000;
    usize max_reply_bytes = 1'500;
    /// Safety limits (characters) per planner style, about twice what the
    /// prompt asks for: cutting a reply short leaves her mid-thought, so
    /// these only catch runaways. Whole sentences are kept where possible.
    usize short_reply_chars = 40;
    usize normal_reply_chars = 80;
    usize long_reply_chars = 300;
    /// Local time for chat logs and the daily budget boundary (UTC+8).
    i32 utc_offset_minutes = 480;
};

struct ArchiveConfig {
    /// Copy media of allowed chats into <data_dir>/media.
    bool enabled = true;
    u64 max_file_bytes = 50ull * 1024 * 1024;
    /// NapCat's QQ data directory inside its container and where that
    /// directory is bind-mounted on the host (paths from get_file are mapped).
    std::string container_qq_dir = "/app/.config/QQ";
    std::string host_qq_dir = "~/services/napcat/qq";
};

struct LearningConfig {
    /// Learn expression patterns from group chat. Every group the account is
    /// in may feed it (read-only); only allowed groups ever get replies.
    bool enabled = true;
    /// Groups never learned from.
    std::vector<i64> exclude_groups;
    /// Lines buffered per group before one learning call.
    usize batch_messages = 30;
    /// Patterns beyond this are dropped (lowest count first); 0 keeps all.
    /// A pattern is a short row, so the pool is allowed to grow.
    usize max_expressions = 0;
    /// Patterns sampled (weighted by count) for the selector per reply.
    usize candidates = 10;
};

struct MemoryConfig {
    /// Searchable history of the allowed chats; needs the local embedding
    /// service (deployment/embedding).
    bool enabled = false;
    std::string embedding_url = "http://127.0.0.1:8090";
    /// Stored with every vector; changing it drops and rebuilds all chunks.
    std::string embedding_model = "Qwen3-Embedding-0.6B";
    /// Prepended to search queries in Qwen3-Embedding's instruction format;
    /// empty sends the query as is.
    std::string query_instruction = "Given a question about a group chat, retrieve the chat messages that answer it";
    /// Messages per chunk, and the pause that starts a new chunk.
    usize chunk_messages = 6;
    i64 chunk_gap_seconds = 1800;
    /// Chunks returned per search, and the least similarity that counts.
    usize search_results = 5;
    double min_score = 0.35;
    /// Searches the planner may run before deciding.
    usize max_searches = 2;
};

struct Config {
    OneBotConfig onebot;
    ModelConfig model;
    ChatConfig chat;
    ArchiveConfig archive;
    LearningConfig learning;
    MemoryConfig memory;
    /// Database and persona location; empty means the platform default.
    std::string data_dir;
    std::string log_level = "info";
};

/// Reads JSON (comments allowed). Unknown keys are errors so typos surface.
/// Secret fields of the form "$NAME" are replaced from the environment; a
/// literal secret is only accepted when the file is private to its owner.
/// `resolve_secrets = false` leaves every secret empty and skips the checks;
/// for inspection tools that never call a model.
Result<Config> load_config(const std::filesystem::path &path, bool resolve_secrets = true);

/// Parses config text without touching the file system or environment.
Result<Config> parse_config(std::string_view text);

/// "$NAME" -> value of NAME (error when unset or empty); anything else as-is.
Result<std::string> resolve_secret(std::string_view value, std::string_view field);

/// $XDG_DATA_HOME/lilia or ~/.local/share/lilia; %LOCALAPPDATA%/lilia on Windows.
std::filesystem::path default_data_dir();

/// ~/.config/lilia/config.json (XDG); %APPDATA%/lilia/config.json on Windows.
std::filesystem::path default_config_path();

} // namespace lilia
