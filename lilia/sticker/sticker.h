#pragma once

#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/async/runtime/task.h>
#include <lighter/http/client.h>

#include <lilia/config.h>
#include <lilia/onebot/types.h>
#include <lilia/provider/history.h>
#include <lilia/provider/provider.h>
#include <lilia/sticker/usage.h>
#include <lilia/store/store.h>

namespace lilia::sticker {

// Sticker learning, after MaiBot's emoji system: stickers posted in any group
// the account is in are downloaded, deduplicated by content hash and counted;
// a vision model describes and tags each (within an hourly budget; animated
// ones as a grid of their frames for models that read only the first) and
// judges whether it is fit to send. Fit stickers fill a capped pool; a popular spare
// takes the place of the weakest member. What a sticker is used for is learned
// from the chats it turns up in (sticker/usage.h): the lines around each
// sighting are kept as an anonymised snippet, like jargon evidence, and the
// helper model re-infers the usage as snippets accumulate. No other text from
// other groups is stored; the recent lines stay in memory.

struct Tagged {
    std::string description;
    std::string tags; ///< "无语、震惊"
    bool usable = false;
};

/// Reads the three-line "描述/标签/适合" answer; nullopt when it does not
/// follow the format.
std::optional<Tagged> parse_tagging(std::string_view answer);

/// Download URLs of the sticker segments (images with sub_type 1, mface).
std::vector<std::string> sticker_urls(const onebot::Message &message);

/// Up to `count` stickers sampled without replacement, weighted by
/// seen + 2 * used + 1 so popular ones come up more.
std::vector<store::Sticker> sample(std::vector<store::Sticker> pool, usize count, std::mt19937 &rng);

/// What a sticker is embedded as: its tags and description.
std::string embedding_text(const store::Sticker &sticker);

/// "用法：…", its learned usage as embedded; empty when none is known.
std::string usage_text(const store::Sticker &sticker);

/// The retrieval inputs of a sticker: embedding_text(), usage_text() when
/// `usage` and known, and its picture (under `data_dir`) when `images`. A
/// search scores the best of them.
std::vector<std::string> embedding_inputs(const store::Sticker &sticker, const std::filesystem::path &data_dir, bool images, bool usage);

/// The words a lexical search matches: tags, description and, when
/// `usage`, the learned usage.
std::string lexical_text(const store::Sticker &sticker, bool usage);

/// Without embeddings: stickers whose tags or description contain words of
/// `feeling` ("无语、震惊"), most matched and most popular first, then a
/// weighted sample to fill up to `count`.
std::vector<store::Sticker> match(std::vector<store::Sticker> pool, std::string_view feeling, usize count, std::mt19937 &rng);

/// "s12": how the planner names a sticker.
std::string handle(const store::Sticker &sticker);

/// "s12", "#s12" or "12" -> 12.
std::optional<i64> parse_handle(std::string_view text);

/// "s12 标签：无语、震惊｜画面：……｜常用于：……" lines (the usage when
/// `usage` and known); stickers she sent within the last day are marked so
/// she does not keep sending the same one.
std::string render_candidates(const std::vector<store::Sticker> &candidates, i64 now, bool usage = true);

/// A message that is a single sticker, as a segment to send again (复读):
/// an image sticker by its URL, an mface as it came; nullopt otherwise.
std::optional<onebot::Segment> resend_segment(const onebot::Message &message);

struct Collector {
    using Spawn = std::move_only_function<void(lighter::Task<>)>;

    Collector(const StickerConfig &config, std::filesystem::path data_dir, store::Store &store, Spawn spawn);

    /// Every group message: collects the stickers in it, and keeps the line
    /// (memory only) as context around sightings.
    void observe(const onebot::MessageEvent &message, i64 self_id);

    /// Her own line in a group, as context around sightings.
    void observe_own_text(i64 group, std::string text);

    /// Her own sticker in a group: a sighting too, so the reactions it gets
    /// become evidence.
    void observe_own(i64 group, i64 sticker_id);

    /// Closes the sightings whose window has passed: stores their snippets and
    /// starts usage inference when one is due.
    void settle_sightings();

    /// Downloads, counts and (when new) tags a sticker; sets `found` to its id.
    lighter::Task<> collect(std::string url, std::string context, std::shared_ptr<std::optional<i64>> found = nullptr);

    /// What `sticker` is used for, from its snippets, stored (helper model).
    lighter::Task<> infer_usage(store::Sticker sticker);

    /// One vision call: the picture (an animated GIF as a grid of its frames
    /// when `frames`)
    /// and the tagging prompt; nullopt when the call fails or the answer is off
    /// the format. Records the call.
    lighter::Task<std::optional<Tagged>> describe(const store::Sticker &sticker, const std::string &bytes, std::string context);

    /// Describes and tags `sticker` within the hourly and background budgets,
    /// then files it into the pool.
    lighter::Task<> tag(store::Sticker sticker, std::string bytes, std::string context);

    /// Registers `sticker` if the pool has room; otherwise it becomes a spare
    /// unless it has been seen more than the weakest registered one.
    void file_into_pool(const store::Sticker &sticker);

    /// Reads a sticker's bytes from the data directory.
    std::optional<std::string> read(const store::Sticker &sticker) const;

    std::move_only_function<i64()> now;
    const StickerConfig &config;
    std::filesystem::path data_dir;
    store::Store &store;
    Spawn spawn;
    lighter::http::Client http;
    std::optional<provider::Provider> vision;
    std::string vision_model;
    /// learning.daily_token_budget: background tokens in 24 h (0: no limit).
    u64 background_budget = 0;
    /// GIFs as grids of their frames (model.vision_gif_frames), and for
    /// experiments a tagging prompt in place of prompts/sticker.md.
    bool frames = false;
    std::string prompt;
    /// Usage inference: the helper model, its prompt, her name for [SELF].
    std::optional<provider::Provider> helper;
    std::string helper_model;
    std::string usage_prompt;
    std::string self_name;

    /// A sticker seen in a group, waiting for the lines after it.
    struct Sighting {
        i64 group = 0;
        i64 time = 0;
        bool own = false;
        std::vector<ContextLine> before;
        ContextLine line;
        std::vector<ContextLine> after;
        std::shared_ptr<std::optional<i64>> sticker_id;
    };
    std::map<i64, std::deque<ContextLine>> recent; ///< per group, memory only
    std::vector<Sighting> sightings;
    std::set<i64> inferring;
    i64 hour = 0;
    usize downloads_this_hour = 0;
    usize descriptions_this_hour = 0;
    usize inferences_this_hour = 0;

private:
    void roll_hour();
    void remember(i64 group, ContextLine line);
    void close(const Sighting &sighting);
    void consider_inference(const store::Sticker &sticker);
    bool within_budget();
};

} // namespace lilia::sticker
