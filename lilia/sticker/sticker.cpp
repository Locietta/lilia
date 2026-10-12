#include "sticker.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <format>
#include <fstream>
#include <sstream>

#include <glaze/json.hpp>

#include <lighter/async/runtime/timeout.h>
#include <lighter/http/bound_client.h>

#include <xmake/bin2obj/sticker_md.hpp>

#include <lilia/archive/archiver.h>
#include <lilia/chat/context.h>
#include <lilia/chat/delivery.h>
#include <lilia/image/frames.h>
#include <lilia/log.h>
#include <lilia/net/codec.h>
#include <lilia/onebot/message.h>
#include <lilia/retrieval/retriever.h>

namespace lilia::sticker {

using lighter::Task;

namespace {

/// The line before a sticker, shown to the tagging call (memory only).
constexpr usize k_context_code_points = 60;
constexpr std::chrono::seconds k_vision_deadline{60};

/// with_timeout takes lighter's error type; hand the outcome through as a value.
Task<Result<provider::ProviderCallCompletion>> settle(lighter::Task<provider::ProviderCallCompletion, Error> call) {
    auto outcome = co_await std::move(call);
    co_return outcome;
}

std::string_view trim(std::string_view text) {
    constexpr std::string_view k_space = " \t\r\n";
    const auto first = text.find_first_not_of(k_space);
    if (first == std::string_view::npos) return {};
    return text.substr(first, text.find_last_not_of(k_space) - first + 1);
}

/// The value after "key：" (or "key:") on one of `answer`'s lines.
std::optional<std::string> line_value(std::string_view answer, std::string_view key) {
    while (!answer.empty()) {
        const auto newline = answer.find('\n');
        auto line = trim(answer.substr(0, newline));
        answer = newline == std::string_view::npos ? std::string_view{} : answer.substr(newline + 1);
        if (!line.starts_with(key)) continue;
        line.remove_prefix(key.size());
        if (line.starts_with("：")) {
            line.remove_prefix(std::string_view("：").size());
        } else if (line.starts_with(":")) {
            line.remove_prefix(1);
        } else {
            continue;
        }
        return std::string(trim(line));
    }
    return std::nullopt;
}

std::string_view prompt_override_or(const std::string &override, std::string_view built_in) {
    return override.empty() ? built_in : std::string_view(override);
}

std::string sha1_hex(std::string_view bytes) {
    static constexpr std::string_view k_hex = "0123456789abcdef";
    std::string out;
    for (const u8 byte : net::sha1(bytes)) {
        out.push_back(k_hex[byte >> 4]);
        out.push_back(k_hex[byte & 0xf]);
    }
    return out;
}

} // namespace

std::optional<Tagged> parse_tagging(std::string_view answer) {
    const auto description = line_value(answer, "描述");
    const auto tags = line_value(answer, "标签");
    const auto usable = line_value(answer, "适合");
    if (!description || !tags || !usable || description->empty()) return std::nullopt;
    return Tagged{.description = *description, .tags = *tags, .usable = usable->starts_with("是")};
}

std::vector<std::string> sticker_urls(const onebot::Message &message) {
    std::vector<std::string> urls;
    for (const auto &segment : message) {
        const bool sticker =
            segment.type == "mface" || (segment.type == "image" && onebot::data_integer(segment, "sub_type").value_or(0) == 1);
        if (!sticker) continue;
        if (auto url = onebot::data_string(segment, "url"); url && (url->starts_with("http://") || url->starts_with("https://"))) {
            urls.push_back(*std::move(url));
        }
    }
    return urls;
}

std::vector<store::Sticker> sample(std::vector<store::Sticker> pool, usize count, std::mt19937 &rng) {
    std::vector<store::Sticker> picked;
    std::vector<double> weights;
    for (const auto &sticker : pool) weights.push_back(static_cast<double>(sticker.seen + 2 * sticker.used + 1));
    while (picked.size() < count && picked.size() < pool.size()) {
        std::discrete_distribution<usize> pick(weights.begin(), weights.end());
        const auto index = pick(rng);
        picked.push_back(pool[index]);
        weights[index] = 0;
    }
    return picked;
}

std::string embedding_text(const store::Sticker &sticker) { return std::format("{}：{}", sticker.tags, sticker.description); }

std::vector<std::string> embedding_inputs(const store::Sticker &sticker, const std::filesystem::path &data_dir, bool images) {
    std::vector<std::string> out{embedding_text(sticker)};
    if (images) out.push_back(retrieval::image_input(data_dir / sticker.path));
    return out;
}

std::vector<store::Sticker> match(std::vector<store::Sticker> pool, std::string_view feeling, usize count, std::mt19937 &rng) {
    std::vector<std::string_view> words;
    for (usize i = 0; i < feeling.size();) {
        // Split at ASCII separators and the common full-width ones.
        static constexpr std::array<std::string_view, 8> k_separators{" ", ",", "/", ":", "、", "，", "：", "；"};
        usize cut = feeling.size();
        usize width = 0;
        for (const auto separator : k_separators) {
            if (const auto at = feeling.find(separator, i); at < cut) {
                cut = at;
                width = separator.size();
            }
        }
        if (cut > i) words.push_back(feeling.substr(i, cut - i));
        i = cut + width;
    }
    std::vector<std::pair<usize, store::Sticker>> hits;
    std::vector<store::Sticker> rest;
    for (auto &sticker : pool) {
        const auto found = std::ranges::count_if(
            words, [&](std::string_view word) { return sticker.tags.contains(word) || sticker.description.contains(word); });
        if (found > 0) {
            hits.emplace_back(static_cast<usize>(found), std::move(sticker));
        } else {
            rest.push_back(std::move(sticker));
        }
    }
    std::ranges::stable_sort(hits, [](const auto &a, const auto &b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second.seen + 2 * a.second.used > b.second.seen + 2 * b.second.used;
    });
    std::vector<store::Sticker> out;
    for (usize i = 0; i < hits.size() && out.size() < count; ++i) out.push_back(std::move(hits[i].second));
    for (auto &sticker : sample(std::move(rest), count - out.size(), rng)) out.push_back(std::move(sticker));
    return out;
}

std::string handle(const store::Sticker &sticker) { return std::format("s{}", sticker.id); }

std::optional<i64> parse_handle(std::string_view text) {
    text = trim(text);
    if (text.starts_with('#')) text.remove_prefix(1);
    if (text.starts_with('s') || text.starts_with('S')) text.remove_prefix(1);
    i64 id = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), id);
    if (ec != std::errc{} || end != text.data() + text.size()) return std::nullopt;
    return id;
}

std::string render_candidates(const std::vector<store::Sticker> &candidates, i64 now) {
    constexpr i64 k_recent_seconds = 86'400;
    std::string out;
    for (const auto &sticker : candidates) {
        out += std::format("{} 标签：{}｜画面：{}{}\n", handle(sticker), sticker.tags, sticker.description,
                           sticker.last_used > 0 && now - sticker.last_used < k_recent_seconds ? "（她一天内发过）" : "");
    }
    return out;
}

std::optional<onebot::Segment> resend_segment(const onebot::Message &message) {
    if (message.size() != 1) return std::nullopt;
    const auto &segment = message.front();
    if (segment.type == "mface") return segment;
    if (segment.type != "image" || onebot::data_integer(segment, "sub_type").value_or(0) != 1) return std::nullopt;
    const auto url = onebot::data_string(segment, "url");
    if (!url || !(url->starts_with("http://") || url->starts_with("https://"))) return std::nullopt;
    onebot::Segment again{.type = "image"};
    again.data["file"] = *url;
    again.data["sub_type"] = i64{1};
    again.data["summary"] = "[动画表情]";
    return again;
}

Collector::Collector(const StickerConfig &config, std::filesystem::path data_dir, store::Store &store, Spawn spawn)
    : now([] { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }),
      config(config), data_dir(std::move(data_dir)), store(store), spawn(std::move(spawn)) {}

void Collector::observe(const onebot::MessageEvent &message, i64 self_id) {
    if (!config.enabled || !message.is_group() || message.user_id == self_id) return;
    const i64 group = *message.group_id;
    if (std::ranges::contains(config.exclude_groups, group)) return;
    for (auto &url : sticker_urls(message.message)) spawn(collect(std::move(url), last_text[group]));
    // Remember the line (memory only) as context for a sticker that follows.
    if (auto text = std::string(trim(onebot::text_only(message.message))); !text.empty()) {
        last_text[group] = chat::count_code_points(text) > k_context_code_points ? std::string() : std::move(text);
    }
}

Task<> Collector::collect(std::string url, std::string context) {
    const i64 current_hour = now() / 3600;
    if (current_hour != hour) {
        hour = current_hour;
        downloads_this_hour = 0;
        descriptions_this_hour = 0;
    }
    if (downloads_this_hour >= config.downloads_per_hour) co_return;
    ++downloads_this_hour;

    auto request = http.on().get(url);
    request.timeout(std::chrono::seconds(30));
    request.curl_option(CURLOPT_FOLLOWLOCATION, 1L);
    request.curl_option(CURLOPT_MAXFILESIZE_LARGE, static_cast<curl_off_t>(config.max_bytes));
    auto response = co_await std::move(request).send();
    if (response.has_error() || !response->ok() || response->body.size() > config.max_bytes) co_return;
    std::string bytes(response->text());
    const auto sha1 = sha1_hex(bytes);

    auto known = store.sticker_by_hash(sha1);
    if (known.has_error()) co_return;
    if (*known) {
        auto sticker = **known;
        std::ignore = store.sticker_seen(sticker.id, now());
        ++sticker.seen;
        if (sticker.status == "pending") {
            co_await tag(std::move(sticker), std::move(bytes), std::move(context));
        } else if (sticker.status == "spare") {
            file_into_pool(sticker);
        }
        co_return;
    }

    const auto extension = archive::detect_extension(bytes, archive::MediaKind::STICKER, "");
    const auto media_type = archive::image_media_type(extension);
    if (media_type.empty()) co_return;
    const auto relative = std::filesystem::path("stickers") / (sha1 + extension);
    std::error_code ec;
    std::filesystem::create_directories(data_dir / "stickers", ec);
    {
        std::ofstream file(data_dir / relative, std::ios::binary | std::ios::trunc);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!file) co_return;
    }
    store::Sticker sticker{
        .sha1 = sha1,
        .path = relative.generic_string(),
        .media_type = std::string(media_type),
        .bytes = static_cast<i64>(bytes.size()),
        .first_seen = now(),
        .last_seen = now(),
    };
    auto id = store.add_sticker(sticker);
    if (id.has_error()) co_return;
    sticker.id = *id;
    co_await tag(std::move(sticker), std::move(bytes), std::move(context));
}

Task<std::optional<Tagged>> Collector::describe(const store::Sticker &sticker, const std::string &bytes, std::string context) {
    if (!vision) co_return std::nullopt;
    // For vision models that read only a GIF's first frame: a grid of frames.
    const auto grid = frames && sticker.media_type == "image/gif" ? image::frame_grid(bytes) : std::nullopt;
    const auto text = chat::fill_template(
        prompt_override_or(prompt, {reinterpret_cast<const char *>(xmake::sticker_md.data()), xmake::sticker_md.size()}),
        {{"frames", grid ?
                        std::format("它是一个动图：这张图是从动图里按时间顺序抽出的 {} 帧，从左到右、从上到下排列，不是几张不同的表情包。",
                                    grid->frames) :
                        std::string()},
         {"context", context.empty() ? std::string() : std::format("它是在这句话之后发的：「{}」（只帮你理解用法）。", context)}});
    std::vector<provider::Part> parts;
    parts.push_back(grid ? provider::ImagePart{.media_type = "image/jpeg", .data = net::base64_encode(grid->jpeg)} :
                           provider::ImagePart{.media_type = sticker.media_type, .data = net::base64_encode(bytes)});
    parts.push_back(provider::TextPart{.text = text});
    provider::History history;
    history.push_back({.role = provider::Role::USER, .parts = std::move(parts)});
    std::string answer;
    const provider::StreamCallbacks callbacks{
        .on_item_completed =
            [&answer](const provider::OutputItem &item) {
                if (const auto *message = std::get_if<provider::AssistantMessageItem>(&item)) {
                    for (const auto &part : message->parts) answer += part.text;
                }
            },
    };
    const i64 started = now();
    auto settled = co_await lighter::with_timeout(settle((*vision)->complete(history, {}, callbacks)), k_vision_deadline);
    auto completion = settled.has_error() ? Result<provider::ProviderCallCompletion>(lighter::outcome_error(
                                                Error::api("timeout", "sticker tagging exceeded its deadline", "", true))) :
                                            *std::move(settled);
    store::LlmCall call{.time = started, .chat = {.type = "sticker", .id = sticker.id}, .model = vision_model};
    if (completion.has_error()) {
        call.error = completion.error().message();
    } else {
        call.input_tokens = completion->usage.input_tokens;
        call.output_tokens = completion->usage.output_tokens;
        call.request_id = completion->request_id;
    }
    std::ignore = store.add_llm_call(call);
    if (completion.has_error()) {
        log::warn("sticker: tagging failed: {}", call.error);
        co_return std::nullopt;
    }
    auto tagged = parse_tagging(answer);
    if (!tagged) log::warn("sticker: unexpected tagging answer: {}", answer);
    co_return tagged;
}

Task<> Collector::tag(store::Sticker sticker, std::string bytes, std::string context) {
    if (!vision || descriptions_this_hour >= config.descriptions_per_hour) co_return; // stays pending; tagged when seen again
    if (background_budget > 0) {
        auto used = store.tokens_since(now() - 86'400, store::TokenScope::BACKGROUND);
        if (used.has_value() && *used >= background_budget) co_return; // likewise
    }
    ++descriptions_this_hour;
    const auto tagged = co_await describe(sticker, bytes, std::move(context));
    if (!tagged) co_return;
    sticker.description = tagged->description;
    sticker.tags = tagged->tags;
    sticker.status = tagged->usable ? "spare" : "rejected";
    std::ignore = store.describe_sticker(sticker.id, sticker.description, sticker.tags, sticker.status);
    log::info("sticker: {} [{}] {}", sticker.status == "rejected" ? "rejected" : "tagged", sticker.tags, sticker.description);
    if (tagged->usable) {
        file_into_pool(sticker);
    } else {
        // Kept as a row (so it is not tagged again), but the image goes.
        std::error_code ec;
        std::filesystem::remove(data_dir / sticker.path, ec);
    }
}

void Collector::file_into_pool(const store::Sticker &sticker) {
    auto registered = store.stickers("registered");
    if (registered.has_error()) return;
    if (registered->size() < config.max_registered) {
        std::ignore = store.set_sticker_status(sticker.id, "registered");
        return;
    }
    // The weakest member: never sent, least seen, longest unseen.
    const auto weakest = std::ranges::min_element(*registered, [](const store::Sticker &a, const store::Sticker &b) {
        return std::tuple(a.used, a.seen, a.last_seen) < std::tuple(b.used, b.seen, b.last_seen);
    });
    if (weakest != registered->end() && sticker.seen > weakest->seen && weakest->used == 0) {
        std::ignore = store.set_sticker_status(weakest->id, "spare");
        std::ignore = store.set_sticker_status(sticker.id, "registered");
    }
}

std::optional<std::string> Collector::read(const store::Sticker &sticker) const {
    std::ifstream file(data_dir / sticker.path, std::ios::binary);
    if (!file) return std::nullopt;
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

} // namespace lilia::sticker
