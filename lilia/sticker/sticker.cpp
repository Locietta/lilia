#include "sticker.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <sstream>

#include <glaze/json.hpp>

#include <lighter/http/bound_client.h>

#include <xmake/bin2obj/sticker_md.hpp>

#include <lilia/archive/archiver.h>
#include <lilia/chat/context.h>
#include <lilia/chat/delivery.h>
#include <lilia/log.h>
#include <lilia/net/codec.h>
#include <lilia/onebot/message.h>

namespace lilia::sticker {

using lighter::Task;

namespace {

/// The line before a sticker, shown to the tagging call (memory only).
constexpr usize k_context_code_points = 60;

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

std::string render_candidates(const std::vector<store::Sticker> &candidates) {
    std::string out;
    for (usize i = 0; i < candidates.size(); ++i) {
        out += std::format("{}. 标签：{}｜画面：{}\n", i + 1, candidates[i].tags, candidates[i].description);
    }
    return out;
}

PickRequest build_pick_request(std::string_view prompt, const std::vector<std::string> &recent, std::string_view intent,
                               const std::vector<store::Sticker> &candidates) {
    PickRequest request;
    provider::append_system(request.history, std::string(prompt));
    std::string user = "最近的聊天：\n";
    for (const auto &line : recent) user += line + "\n";
    user += std::format("\n她想表达的意思：{}\n\n可以发的表情包：\n{}", intent, render_candidates(candidates));
    provider::append_user(request.history, std::move(user));
    glz::generic item_schema;
    std::ignore = glz::read_json(item_schema, R"({"type": "string"})");
    provider::SchemaProperty ids{.type = "array", .description = "选中的编号，最多一个；不挑就为空"};
    ids.items.emplace(std::move(item_schema));
    request.tools.push_back({
        .name = "pick_sticker",
        .description = "提交要发的表情包编号",
        .input_schema = {.properties = {{"ids", std::move(ids)}}, .required = {"ids"}},
    });
    return request;
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

Task<> Collector::tag(store::Sticker sticker, std::string bytes, std::string context) {
    if (!vision || descriptions_this_hour >= config.descriptions_per_hour) co_return; // stays pending; tagged when seen again
    ++descriptions_this_hour;
    const auto prompt = chat::fill_template(
        {reinterpret_cast<const char *>(xmake::sticker_md.data()), xmake::sticker_md.size()},
        {{"context", context.empty() ? std::string() : std::format("它是在这句话之后发的：「{}」（只帮你理解用法）。", context)}});
    provider::History history;
    history.push_back({.role = provider::Role::USER,
                       .parts = {provider::ImagePart{.media_type = sticker.media_type, .data = net::base64_encode(bytes)},
                                 provider::TextPart{.text = prompt}}});
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
    auto completion = co_await (*vision)->complete(history, {}, callbacks);
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
        co_return;
    }
    const auto tagged = parse_tagging(answer);
    if (!tagged) {
        log::warn("sticker: unexpected tagging answer: {}", answer);
        co_return;
    }
    sticker.description = tagged->description;
    sticker.tags = tagged->tags;
    sticker.status = tagged->usable ? "spare" : "rejected";
    std::ignore = store.describe_sticker(sticker.id, sticker.description, sticker.tags, sticker.status);
    log::info("sticker: {} [{}] {}", sticker.status == "rejected" ? "rejected" : "tagged", sticker.tags, sticker.description);
    if (tagged->usable) file_into_pool(sticker);
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
