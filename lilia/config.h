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
/// A backup model for the writer or the planner, tried when the main one fails.
struct ModelFallback {
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
    /// Learning, memory summaries and search, on `base_url`; empty means
    /// `planner_model`. Set it when the planner has its own endpoint.
    std::string helper_model;
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
    /// means `base_url`. Search, vision and learning stay on `base_url`
    /// (see helper_model).
    std::string planner_base_url;
    std::string planner_api_key; ///< secret
    bool planner_bearer = false;
    std::string planner_thinking_off; ///< like writer_thinking_off
    /// Force a tool call with tool_choice "any". Off by default: the planner
    /// writes a short analysis before its tool call (MaiBot's), which forcing
    /// would prevent; an answer without a call counts as staying silent.
    bool planner_force_tool = false;
    /// Tried in order when a planner call fails (outage, rate limit, a used-up
    /// subscription window). They get the planner's tool forcing and limits.
    std::vector<ModelFallback> planner_fallbacks;
    /// Deadlines (seconds) for one model call, retries included. Past them a
    /// call counts as failed, so a hung request cannot leave her silent.
    u32 timeout_seconds = 30;
    u32 writer_timeout_seconds = 40;
    u32 search_timeout_seconds = 90;
    /// Tried in order when the writer call fails (outage, rate limit) or
    /// comes back empty.
    std::vector<ModelFallback> writer_fallbacks;
    /// Writer sampling temperature; unset uses the provider default.
    std::optional<double> writer_temperature;
    /// Describes images and stickers (on `base_url`); empty disables it.
    std::string vision_model = "deepseek-flash";
    /// Send an animated GIF as one grid of its frames, for vision models that
    /// read only the first frame. DeepSeek reads GIFs whole (2026-10-12: it
    /// described motion without the grid, and the grid scored no better).
    bool vision_gif_frames = false;
    /// Offer the provider-hosted web_search server tool. A searched answer
    /// costs tens of thousands of input tokens with DeepSeek.
    bool web_search = true;

    const std::string &helper() const { return helper_model.empty() ? planner_model : helper_model; }
};

/// Homophone typos in casual replies (MaiBot's chinese_typo, same defaults).
struct TypoConfig {
    bool enabled = true;
    /// Chance a single-character word is mistyped; inside an n-character
    /// word each character gets error_rate * 0.7^(n-1).
    double error_rate = 0.01;
    /// Homophones less frequent than this (character frequency, 0..1000)
    /// are never used.
    double min_freq = 9;
    /// Chance the homophones of another tone are considered too.
    double tone_error_rate = 0.1;
    /// Chance a whole word is swapped for a homophone word.
    double word_replace_rate = 0.006;
    /// The follow-up correction quotes the mistyped message.
    bool correction_quote = true;
};

/// Her day (live only; replay and tests ignore it): asleep at night, a few
/// busy stretches, and a moment to read before reacting.
struct ScheduleConfig {
    bool enabled = false;
    /// Local "HH:MM". Asleep she does nothing; whoever called her is answered
    /// once she is up.
    std::string sleep_start = "01:30";
    std::string sleep_end = "07:00";
    /// Busy stretches per day, at random times between 09:00 and 22:00 (the
    /// same all day). Busy, she does not chime in; a call is answered late.
    u32 busy_blocks = 2;
    u32 busy_min_minutes = 20;
    u32 busy_max_minutes = 60;
    u32 busy_reply_min_seconds = 60;
    u32 busy_reply_max_seconds = 300;
    /// Before reacting: called, and otherwise.
    u32 read_called_min_seconds = 1;
    u32 read_called_max_seconds = 8;
    u32 read_min_seconds = 3;
    u32 read_max_seconds = 20;
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
    /// 0..1, how often the planner looks: every ceil(1/talk_value) new
    /// messages (1, MaiBot's default: every batch); 0 only when called. It
    /// always sees everything that came in; it decides whether she speaks.
    double talk_value = 1;
    /// Wait for a burst of messages to end before deciding.
    u32 debounce_seconds = 3;
    /// What the planner hands the writer: "intent" (a short stance to voice)
    /// or "reference" (notes the writer may use or ignore, like MaiBot).
    std::string handoff = "intent";
    /// Typing-delay speed for multi-message replies; 0 sends immediately.
    double typing_speed = 1.0;
    TypoConfig typos;
    ScheduleConfig schedule;
    /// How long a turn waits for new images to be described.
    u32 image_wait_seconds = 8;
    /// Input plus output tokens per local calendar day across all chats
    /// (calls made for them; learning has its own budget).
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
    /// Others' lines buffered per group before one learning batch (her own
    /// lines ride along as [SELF] context), at least this many seconds apart
    /// per group, with at most `max_concurrent` batches running at once.
    usize batch_messages = 30;
    i64 min_interval_seconds = 30;
    usize max_concurrent = 3;
    /// Patterns beyond this are dropped (lowest count first); 0 keeps all.
    /// A pattern is a short row, so the pool is allowed to grow.
    usize max_expressions = 0;
    /// Candidates the selector sees per reply: retrieved by embedding against
    /// the planner's intent (with MMR for variety), or a weighted sample when
    /// there are no embeddings.
    usize candidates = 30;
    /// Patterns the selector may pick for one reply (MaiBot's is 5;
    /// prompts/select.md asks for at most 3).
    usize max_selected = 3;
    /// Use only patterns a person has checked (`lilia expressions --check`).
    bool checked_only = false;
    /// Chats that share learned patterns: each inner list is a sharing group,
    /// 0 stands for every chat. Empty: everything is shared.
    std::vector<std::vector<i64>> expression_groups;
    /// Slang learning (MaiBot's jargon): extract terms, infer meanings at
    /// count thresholds, show them to the planner when they come up.
    bool jargon = true;
    /// Sharing groups for slang, as for expressions.
    std::vector<std::vector<i64>> jargon_groups;
    /// Evidence snippets kept per term (speakers anonymised).
    usize jargon_evidence = 10;
    /// Terms shown to the planner per turn.
    usize jargon_matches = 10;
    /// Tokens per rolling 24 hours for background work (learning batches and
    /// sticker tagging count). Past it new learning batches wait; replies are
    /// not affected. 0: no limit.
    u64 daily_token_budget = 1'000'000;
};

struct MemoryConfig {
    /// Searchable history of the allowed chats; needs the local embedding
    /// service (deployment/embedding).
    bool enabled = false;
    std::string embedding_url = "http://127.0.0.1:8090";
    /// OpenAI-style embeddings route: /v1/embeddings (llama.cpp) or
    /// /v3/embeddings (OpenVINO Model Server, qvl).
    std::string embedding_path = "/v1/embeddings";
    /// Sent as the request's model and stored with every vector; changing it drops and rebuilds all chunks.
    std::string embedding_model = "Qwen3-Embedding-0.6B";
    /// The task of chat-search queries; empty sends the query as is.
    std::string query_instruction = "Given a question about a group chat, retrieve the chat messages that answer it";
    /// How a query's task reaches the model: "inline" prefixes it in
    /// Qwen3-Embedding's "Instruct: …\nQuery:…" form; "field" sends it as the
    /// request's "instruction", which qvl puts in Qwen3-VL-Embedding's system turn.
    std::string instruction_mode = "inline";
    /// Also embed each sticker's picture (by its file path), for a multimodal
    /// service such as qvl that may read lilia's sticker files.
    bool embed_images = false;
    /// A Cohere-style reranker (qvl's /v3/rerank); empty url: none.
    std::string rerank_url;
    std::string rerank_path = "/v3/rerank";
    /// Sent as the request's model; empty lets the service choose.
    std::string rerank_model;
    /// Sticker searches rerank this many fused candidates down to
    /// stickers.candidates (0: no reranking).
    usize rerank_stickers = 0;
    /// Show the reranker pictures too, not only text (about 2.4x the time).
    bool rerank_images = true;
    /// Messages per chunk, and the pause that starts a new chunk.
    usize chunk_messages = 6;
    i64 chunk_gap_seconds = 1800;
    /// Chunks returned per search, and the least similarity that counts.
    usize search_results = 5;
    double min_score = 0.35;
    /// Searches the planner may run before deciding.
    usize max_searches = 2;
    /// Learn facts about the person she answers (from their own words).
    bool facts = true;
    /// Facts shown per person, and the similarity that counts as a repeat.
    usize facts_per_person = 6;
    /// Measured with the default Qwen3-Embedding-0.6B: paraphrases ~0.86-0.88,
    /// different facts ("学吉他" vs "学钢琴") ~0.75. Similarities shift with the
    /// model; re-measure this and the scores below when changing it.
    double duplicate_fact_score = 0.82;
    /// Messages per rolling summary (0 turns summaries off).
    usize summary_messages = 36;
    /// Recall: the last messages compared against window summary cues, and
    /// the similarity that brings a summary back (measured on rolling
    /// summaries' cues; window summaries' may want another).
    usize recall_messages = 6;
    /// Measured with the default Qwen3-Embedding-0.6B (tests/replay/recall.jsonl):
    /// unrelated chatter ~0.45, the topic coming back 0.61-0.69.
    double recall_score = 0.6;
};

struct StickerConfig {
    /// Collect stickers from every group (images only, read-only) and let
    /// lilia send registered ones. Needs model.vision_model for tagging.
    bool enabled = false;
    /// Groups never collected from.
    std::vector<i64> exclude_groups;
    /// Registered stickers she may send. Once full, a spare hotter than the
    /// coldest member takes its place; one a person edited is never displaced.
    usize max_registered = 200;
    /// Heat, the popularity that ranks stickers: a sighting adds 1, her send
    /// 2, and it halves every this many days (0: never).
    double half_life_days = 14;
    /// Budgets: new stickers described (vision calls) and downloads per hour.
    usize descriptions_per_hour = 10;
    usize downloads_per_hour = 200;
    u64 max_bytes = 2ull * 1024 * 1024;
    /// Stickers find_sticker shows the planner, and the pause between hers.
    usize candidates = 8;
    i64 cooldown_seconds = 60;
    /// Learn what stickers are used for from the lines around them: snippets
    /// kept per sticker, lines on each side, how long to wait for the lines
    /// after, inference calls per hour, and whether searches use the usage.
    bool usage = true;
    usize usage_evidence = 12;
    usize usage_context_lines = 3;
    i64 usage_after_seconds = 180;
    usize usage_inferences_per_hour = 20;
    bool usage_in_search = true;
};

struct FeedbackConfig {
    /// Judge her chats in hindsight (helper model): whom each line was said
    /// to, how her messages went down; learn an addressee prior from it.
    bool enabled = false;
    /// Show the prior to the planner; off, it is only logged (planner_turns).
    bool hint = false;
    /// Show it before any model is fitted (hand-set weights).
    bool hint_bootstrap = false;
    /// A window is judged this long after her last message in it, or once
    /// this many lines wait; at most this many judge calls an hour.
    i64 judge_after_seconds = 600;
    usize judge_window_lines = 40;
    usize judges_per_hour = 6;
    /// Labels before a model is fitted: pooled, and per group (with this many
    /// lines that were to her); refits after every refit_every new labels.
    usize min_rows_global = 60;
    usize min_rows_group = 80;
    usize min_positives = 10;
    usize refit_every = 20;
    /// Sticker search boost: reward_weight * 6 * (m - 0.5), m the Beta mean of
    /// her sends of it going down well; at most ±3 * reward_weight.
    double reward_weight = 0.02;
};

struct Config {
    OneBotConfig onebot;
    ModelConfig model;
    ChatConfig chat;
    ArchiveConfig archive;
    LearningConfig learning;
    MemoryConfig memory;
    StickerConfig stickers;
    FeedbackConfig feedback;
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
