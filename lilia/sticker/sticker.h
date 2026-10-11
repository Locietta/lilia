#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/async/runtime/task.h>
#include <lighter/http/client.h>

#include <lilia/config.h>
#include <lilia/onebot/types.h>
#include <lilia/provider/history.h>
#include <lilia/provider/provider.h>
#include <lilia/store/store.h>

namespace lilia::sticker {

// Sticker learning, after MaiBot's emoji system: stickers posted in any group
// the account is in are downloaded, deduplicated by content hash and counted;
// a vision model describes and tags each (within an hourly budget) and judges
// whether it is fit to send. Fit stickers fill a capped pool; a popular spare
// takes the place of the weakest member. Other groups' text is never stored:
// the line before a sticker only helps the tagging call, in memory.

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

/// Without embeddings: stickers whose tags or description contain words of
/// `feeling` ("无语、震惊"), most matched and most popular first, then a
/// weighted sample to fill up to `count`.
std::vector<store::Sticker> match(std::vector<store::Sticker> pool, std::string_view feeling, usize count, std::mt19937 &rng);

/// "s12": how the planner names a sticker.
std::string handle(const store::Sticker &sticker);

/// "s12", "#s12" or "12" -> 12.
std::optional<i64> parse_handle(std::string_view text);

/// "s12 标签：无语、震惊｜画面：……" lines; stickers she sent within the last
/// day are marked so she does not keep sending the same one.
std::string render_candidates(const std::vector<store::Sticker> &candidates, i64 now);

/// A message that is a single sticker, as a segment to send again (复读):
/// an image sticker by its URL, an mface as it came; nullopt otherwise.
std::optional<onebot::Segment> resend_segment(const onebot::Message &message);

struct Collector {
    using Spawn = std::move_only_function<void(lighter::Task<>)>;

    Collector(const StickerConfig &config, std::filesystem::path data_dir, store::Store &store, Spawn spawn);

    /// Every group message: remembers its text (memory only) and collects
    /// the stickers in it.
    void observe(const onebot::MessageEvent &message, i64 self_id);

    lighter::Task<> collect(std::string url, std::string context);

    /// Describes and tags `sticker`, then files it into the pool.
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
    std::map<i64, std::string> last_text; ///< per group, never written anywhere
    i64 hour = 0;
    usize downloads_this_hour = 0;
    usize descriptions_this_hour = 0;
};

} // namespace lilia::sticker
