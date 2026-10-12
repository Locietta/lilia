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
#include <xmake/bin2obj/sticker_usage_md.hpp>

#include <lilia/archive/archiver.h>
#include <lilia/chat/context.h>
#include <lilia/chat/delivery.h>
#include <lilia/image/frames.h>
#include <lilia/learn/learner.h>
#include <lilia/log.h>
#include <lilia/net/codec.h>
#include <lilia/onebot/message.h>
#include <lilia/retrieval/retriever.h>

namespace lilia::sticker {

using lighter::Task;

namespace {

/// The line before a sticker, shown to the tagging call (memory only).
constexpr usize k_context_code_points = 60;
constexpr i64 k_context_seconds = 120;
constexpr std::chrono::seconds k_vision_deadline{60};
/// Recent lines per group kept for sightings, and how old they may be.
constexpr usize k_recent_lines = 8;
constexpr i64 k_recent_seconds = 600;
/// A sighting whose sticker never got an id (not downloaded) is dropped.
constexpr i64 k_unknown_sticker_seconds = 600;

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

/// One model call about a sticker with a deadline, recorded as chat type
/// "sticker" (the background budget); nullopt when it failed.
Task<std::optional<std::vector<provider::OutputItem>>>
call_model(Collector &collector, provider::Provider &provider, const std::string &model, i64 sticker_id, const provider::History &history,
           const std::vector<provider::ToolDefinition> &tools, std::string_view purpose) {
    std::vector<provider::OutputItem> items;
    const provider::StreamCallbacks callbacks{
        .on_item_completed = [&items](const provider::OutputItem &item) { items.push_back(item); },
    };
    const i64 started = collector.now();
    auto settled = co_await lighter::with_timeout(settle(provider->complete(history, tools, callbacks)), k_vision_deadline);
    auto completion = settled.has_error() ? Result<provider::ProviderCallCompletion>(lighter::outcome_error(
                                                Error::api("timeout", std::format("{} exceeded its deadline", purpose), "", true))) :
                                            *std::move(settled);
    store::LlmCall call{.time = started, .chat = {.type = "sticker", .id = sticker_id}, .model = model};
    if (completion.has_error()) {
        call.error = completion.error().message();
    } else {
        call.input_tokens = completion->usage.input_tokens;
        call.output_tokens = completion->usage.output_tokens;
        call.request_id = completion->request_id;
    }
    std::ignore = collector.store.add_llm_call(call);
    if (completion.has_error()) {
        log::warn("sticker: {} failed: {}", purpose, call.error);
        co_return std::nullopt;
    }
    co_return std::move(items);
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
    for (const auto &sticker : pool)
        weights.push_back(static_cast<double>(sticker.seen + 2 * sticker.used + 1) + std::max(0.0, sticker.reward));
    while (picked.size() < count && picked.size() < pool.size()) {
        std::discrete_distribution<usize> pick(weights.begin(), weights.end());
        const auto index = pick(rng);
        picked.push_back(pool[index]);
        weights[index] = 0;
    }
    return picked;
}

std::string embedding_text(const store::Sticker &sticker) { return std::format("{}：{}", sticker.tags, sticker.description); }

std::string usage_text(const store::Sticker &sticker) { return sticker.usage.empty() ? std::string() : "用法：" + sticker.usage; }

std::vector<std::string> embedding_inputs(const store::Sticker &sticker, const std::filesystem::path &data_dir, bool images, bool usage) {
    std::vector<std::string> out{embedding_text(sticker)};
    if (usage && !sticker.usage.empty()) out.push_back(usage_text(sticker));
    // A missing file would fail the embedding call of every text sent with it.
    if (std::error_code ec; images && std::filesystem::exists(data_dir / sticker.path, ec)) {
        out.push_back(retrieval::image_input(data_dir / sticker.path));
    }
    return out;
}

std::string lexical_text(const store::Sticker &sticker, bool usage) {
    auto text = sticker.tags + " " + sticker.description;
    if (usage && !sticker.usage.empty()) text += " " + sticker.usage;
    return text;
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

std::string render_candidates(const std::vector<store::Sticker> &candidates, i64 now, bool usage) {
    constexpr i64 k_recent_seconds = 86'400;
    std::string out;
    for (const auto &sticker : candidates) {
        out += std::format("{} 标签：{}｜画面：{}{}{}\n", handle(sticker), sticker.tags, sticker.description,
                           !usage || sticker.usage.empty() ? std::string() : "｜常用于：" + sticker.usage,
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
      config(config), data_dir(std::move(data_dir)), store(store), spawn(std::move(spawn)),
      usage_prompt(reinterpret_cast<const char *>(xmake::sticker_usage_md.data()), xmake::sticker_usage_md.size()) {}

void Collector::roll_hour() {
    const i64 current_hour = now() / 3600;
    if (current_hour == hour) return;
    hour = current_hour;
    downloads_this_hour = 0;
    descriptions_this_hour = 0;
    inferences_this_hour = 0;
}

bool Collector::within_budget() {
    if (background_budget == 0) return true;
    auto used = store.tokens_since(now() - 86'400, store::TokenScope::BACKGROUND);
    return !used.has_value() || *used < background_budget;
}

void Collector::observe(const onebot::MessageEvent &message, i64 self_id) {
    if (!config.enabled || !message.is_group() || message.user_id == self_id) return;
    const i64 group = *message.group_id;
    if (std::ranges::contains(config.exclude_groups, group)) return;
    const auto urls = sticker_urls(message.message);
    ContextLine line{.sender = message.user_id,
                     .name = message.sender.card.empty() ? message.sender.nickname : message.sender.card,
                     .text = learn::learnable_text(message.message),
                     .time = now()};
    // The line before, for the tagging call: recent, short, not media.
    std::string context;
    if (const auto &lines = recent[group]; !lines.empty()) {
        const auto &last = lines.back();
        if (!last.self && now() - last.time <= k_context_seconds && !last.text.starts_with('[') &&
            chat::count_code_points(last.text) <= k_context_code_points) {
            context = last.text;
        }
    }
    // Earlier sightings take this line as one after theirs; its own come next.
    auto shown = line;
    if (shown.text.empty()) {
        const bool image = std::ranges::any_of(message.message, [](const onebot::Segment &segment) { return segment.type == "image"; });
        shown.text = !urls.empty() ? "[表情]" : image ? "[图片]" : "";
    }
    remember(group, shown);
    for (auto url : urls) {
        auto found = std::make_shared<std::optional<i64>>();
        if (config.usage) {
            Sighting sighting{.group = group, .time = now(), .line = line, .sticker_id = found};
            const auto &lines = recent[group];
            // The ring already holds this line; take the ones before it.
            const usize available = lines.empty() ? 0 : lines.size() - 1;
            for (usize i = available > config.usage_context_lines ? available - config.usage_context_lines : 0; i < available; ++i) {
                sighting.before.push_back(lines[i]);
            }
            sightings.push_back(std::move(sighting));
        }
        spawn(collect(std::move(url), context, std::move(found)));
    }
    settle_sightings();
}

void Collector::observe_own_text(i64 group, std::string text) {
    if (!config.enabled || text.empty() || std::ranges::contains(config.exclude_groups, group)) return;
    remember(group, {.text = std::move(text), .time = now(), .self = true});
    settle_sightings();
}

void Collector::observe_own(i64 group, i64 sticker_id) {
    if (!config.enabled || !config.usage || std::ranges::contains(config.exclude_groups, group)) return;
    Sighting sighting{.group = group,
                      .time = now(),
                      .own = true,
                      .line = {.time = now(), .self = true},
                      .sticker_id = std::make_shared<std::optional<i64>>(sticker_id)};
    const auto &lines = recent[group];
    for (usize i = lines.size() > config.usage_context_lines ? lines.size() - config.usage_context_lines : 0; i < lines.size(); ++i) {
        sighting.before.push_back(lines[i]);
    }
    remember(group, {.text = "[表情]", .time = now(), .self = true});
    sightings.push_back(std::move(sighting));
    settle_sightings();
}

void Collector::remember(i64 group, ContextLine line) {
    if (line.text.empty()) return;
    for (auto &sighting : sightings) {
        if (sighting.group == group && sighting.after.size() < config.usage_context_lines) sighting.after.push_back(line);
    }
    auto &lines = recent[group];
    lines.push_back(std::move(line));
    while (lines.size() > k_recent_lines || (!lines.empty() && now() - lines.front().time > k_recent_seconds)) lines.pop_front();
}

void Collector::settle_sightings() {
    std::vector<Sighting> open;
    std::vector<Sighting> done;
    for (auto &sighting : sightings) {
        const bool known = sighting.sticker_id && sighting.sticker_id->has_value();
        const bool complete = sighting.after.size() >= config.usage_context_lines || now() - sighting.time >= config.usage_after_seconds;
        if (complete && known) {
            done.push_back(std::move(sighting));
        } else if (!known && now() - sighting.time >= k_unknown_sticker_seconds) {
            // Never downloaded (hourly cap, too large, gone): nothing to file it under.
        } else {
            open.push_back(std::move(sighting));
        }
    }
    sightings = std::move(open);
    for (const auto &sighting : done) close(sighting);
}

void Collector::close(const Sighting &sighting) {
    const i64 id = **sighting.sticker_id;
    auto sticker = store.sticker_by_id(id);
    if (!sticker.has_value() || !*sticker || (*sticker)->status == "rejected") return;
    auto lines = sighting.before;
    lines.push_back(sighting.line);
    lines.insert(lines.end(), sighting.after.begin(), sighting.after.end());
    const auto snippet = usage_snippet(lines, sighting.before.size(), self_name);
    auto updated = store.add_sticker_evidence(id, {.type = "group", .id = sighting.group}, snippet, sighting.own, now(),
                                              std::max<usize>(config.usage_evidence, 1));
    if (updated.has_error()) {
        log::error("cannot store sticker evidence: {}", updated.error().message());
        return;
    }
    consider_inference(*updated);
}

void Collector::consider_inference(const store::Sticker &sticker) {
    if (!helper || !usage_due(sticker.usage_count, sticker.usage_inferred_count)) return;
    if ((sticker.status != "registered" && sticker.status != "spare") || inferring.contains(sticker.id)) return;
    roll_hour();
    // Over a limit it waits for its next snippet, still due then.
    if (inferences_this_hour >= config.usage_inferences_per_hour || !within_budget()) return;
    ++inferences_this_hour;
    inferring.insert(sticker.id);
    spawn(infer_usage(sticker));
}

Task<> Collector::infer_usage(store::Sticker sticker) {
    auto evidence = store.sticker_evidence(sticker.id, std::max<usize>(config.usage_evidence, 1));
    if (evidence.has_error() || evidence->empty() || !helper) {
        inferring.erase(sticker.id);
        co_return;
    }
    std::set<i64> groups;
    for (const auto &snippet : *evidence) groups.insert(snippet.chat.id);
    const auto request = build_usage_request(usage_prompt, sticker, *evidence, groups.size(), self_name);
    const auto items = co_await call_model(*this, *helper, helper_model, sticker.id, request.history, request.tools, "sticker usage");
    inferring.erase(sticker.id);
    if (!items) co_return;
    const auto usage = parse_usage(*items);
    if (!usage) {
        log::warn("sticker: no usage answer for {}", handle(sticker));
        co_return;
    }
    if (usage->no_info) {
        std::ignore = store.set_sticker_usage(sticker.id, std::nullopt, std::nullopt, sticker.usage_count);
        log::info("sticker: {} has no fixed use yet ({} snippets)", handle(sticker), sticker.usage_count);
    } else {
        std::ignore =
            store.set_sticker_usage(sticker.id, usage->usage,
                                    usage->tags.empty() ? std::nullopt : std::optional<std::string_view>(usage->tags), sticker.usage_count);
        log::info("sticker: {} [{}] {} ({} snippets)", handle(sticker), usage->tags, usage->usage, sticker.usage_count);
    }
}

Task<> Collector::collect(std::string url, std::string context, std::shared_ptr<std::optional<i64>> found) {
    roll_hour();
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
        if (found) *found = sticker.id;
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
    if (found) *found = sticker.id;
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
    const auto items = co_await call_model(*this, *vision, vision_model, sticker.id, history, {}, "sticker tagging");
    if (!items) co_return std::nullopt;
    std::string answer;
    for (const auto &item : *items) {
        if (const auto *message = std::get_if<provider::AssistantMessageItem>(&item)) {
            for (const auto &part : message->parts) answer += part.text;
        }
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
