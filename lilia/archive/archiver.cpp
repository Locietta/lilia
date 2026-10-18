#include "archiver.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <format>
#include <fstream>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>

#include <curl/curl.h>
#include <glaze/json.hpp>

#include <lighter/async/runtime/timeout.h>
#include <lighter/http/bound_client.h>

#include <lilia/calls.h>
#include <lilia/log.h>
#include <lilia/net/codec.h>
#include <lilia/onebot/message.h>

namespace lilia::archive {

using lighter::Task;

namespace {

/// Error text; "too large" outcomes are told apart by `too_large`.
struct FetchError {
    std::string detail;
    bool too_large = false;
};

struct Fetched {
    std::string bytes;
    std::string source;
};

using FetchResult = lighter::Outcome<Fetched, FetchError>;

struct FileInfo {
    std::optional<std::string> file;
    std::optional<std::string> url;
    std::optional<std::string> file_name;
};

struct FileUrl {
    std::string url;
};

bool is_http(std::string_view text) { return text.starts_with("http://") || text.starts_with("https://"); }

/// Anthropic's limit for one inline image.
constexpr usize k_max_vision_bytes = 5 * 1024 * 1024;
constexpr usize k_max_description_code_points = 120;

constexpr std::string_view k_vision_prompt =
    "用一两句中文描述这张图，像跟朋友转述一样简洁。是表情包的话说它表达的情绪或者梗；图上有文字就照写出来。不要用 Markdown，不要换行。";

constexpr std::chrono::seconds k_vision_deadline{60};

/// Collapses whitespace and newlines and bounds the length.
std::string one_line(std::string_view text) {
    std::string out;
    bool space = false;
    for (const char c : text) {
        if (c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            space = !out.empty();
            continue;
        }
        if (space) out.push_back(' ');
        space = false;
        out.push_back(c);
    }
    usize kept = 0;
    usize bytes = 0;
    while (bytes < out.size() && kept < k_max_description_code_points) {
        ++bytes;
        while (bytes < out.size() && (static_cast<unsigned char>(out[bytes]) & 0xC0) == 0x80) ++bytes;
        ++kept;
    }
    if (bytes < out.size()) out = out.substr(0, bytes) + "…";
    return out;
}

/// Large media is read, hashed and written this much at a time, with the
/// loop free to serve chats in between (everything runs on one thread).
constexpr usize k_io_chunk = 1 << 20;

/// The SHA-1 of `bytes` as hex, a chunk at a time; `bytes` must outlive it.
Task<std::string> sha1_hex(std::string_view bytes) {
    static constexpr std::string_view k_hex = "0123456789abcdef";
    net::Sha1 hash;
    for (usize at = 0; at < bytes.size(); at += k_io_chunk) {
        hash.update(bytes.substr(at, k_io_chunk));
        if (at + k_io_chunk < bytes.size()) co_await lighter::yield();
    }
    std::string out;
    for (const u8 byte : hash.finish()) {
        out.push_back(k_hex[byte >> 4]);
        out.push_back(k_hex[byte & 0xf]);
    }
    co_return out;
}

FetchResult too_large(u64 bytes, u64 limit) {
    return lighter::outcome_error(
        FetchError{.detail = std::to_string(bytes) + " bytes exceeds the " + std::to_string(limit) + " byte limit", .too_large = true});
}

Task<FetchResult> download(Archiver &archiver, std::string url) {
    const auto limit = archiver.config.max_file_bytes;
    auto request = archiver.http.on().get(url);
    request.timeout(std::chrono::minutes(2));
    request.curl_option(CURLOPT_FOLLOWLOCATION, 1L);
    request.curl_option(CURLOPT_MAXFILESIZE_LARGE, static_cast<curl_off_t>(limit));
    auto response = co_await std::move(request).send();
    if (response.has_error()) co_return lighter::outcome_error(FetchError{.detail = "download failed: " + response.error().message()});
    if (!response->ok())
        co_return lighter::outcome_error(FetchError{.detail = "download returned HTTP " + std::to_string(response->status)});
    if (response->body.size() > limit) co_return too_large(response->body.size(), limit);
    co_return Fetched{.bytes = std::string(response->text()), .source = std::move(url)};
}

Task<FetchResult> read_container_file(const Archiver &archiver, std::string container_path) {
    const auto host = map_container_path(container_path, archiver.config);
    if (!host) co_return lighter::outcome_error(FetchError{.detail = "path outside the shared QQ directory: " + container_path});
    std::error_code ec;
    const auto size = std::filesystem::file_size(*host, ec);
    if (ec) co_return lighter::outcome_error(FetchError{.detail = "cannot stat " + host->string() + ": " + ec.message()});
    if (size > archiver.config.max_file_bytes) co_return too_large(size, archiver.config.max_file_bytes);
    std::ifstream file(*host, std::ios::binary);
    if (!file) co_return lighter::outcome_error(FetchError{.detail = "cannot open " + host->string()});
    std::string bytes;
    bytes.reserve(static_cast<usize>(size));
    std::string chunk(k_io_chunk, '\0');
    while (file) {
        file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        bytes.append(chunk.data(), static_cast<usize>(file.gcount()));
        if (file) co_await lighter::yield();
    }
    if (file.bad()) co_return lighter::outcome_error(FetchError{.detail = "cannot read " + host->string()});
    co_return Fetched{.bytes = std::move(bytes), .source = "napcat:" + container_path};
}

template <typename T>
std::optional<T> parse_data(std::string_view json) {
    T value{};
    if (glz::read<onebot::k_read_opts>(value, json)) return std::nullopt;
    return value;
}

Task<FetchResult> fetch(Archiver &archiver, const std::shared_ptr<onebot::Session> &session, const store::ChatKey &chat,
                        const MediaRef &ref) {
    if (is_http(ref.url)) co_return co_await download(archiver, ref.url);
    if (!ref.path.empty()) co_return co_await read_container_file(archiver, ref.path);
    if (!session || !session->is_open()) co_return lighter::outcome_error(FetchError{.detail = "no OneBot session to resolve the file"});

    if (ref.kind == MediaKind::FILE && chat.type == "group" && !ref.file_id.empty()) {
        auto params = std::format(R"({{"group_id":{},"file_id":{}}})", chat.id, glz::write_json(ref.file_id).value_or("\"\""));
        auto data = co_await session->call("get_group_file_url", std::move(params));
        if (data.has_value()) {
            if (auto resolved = parse_data<FileUrl>(*data); resolved && is_http(resolved->url))
                co_return co_await download(archiver, resolved->url);
        }
    }

    if (!ref.file.empty()) {
        auto params = std::format(R"({{"file":{}}})", glz::write_json(ref.file).value_or("\"\""));
        auto data = co_await session->call("get_file", std::move(params));
        if (data.has_error()) co_return lighter::outcome_error(FetchError{.detail = "get_file failed: " + data.error().message()});
        auto info = parse_data<FileInfo>(*data);
        if (!info) co_return lighter::outcome_error(FetchError{.detail = "get_file returned undecodable data"});
        if (info->file && info->file->starts_with('/')) co_return co_await read_container_file(archiver, *info->file);
        if (info->url && is_http(*info->url)) co_return co_await download(archiver, *info->url);
        co_return lighter::outcome_error(FetchError{.detail = "get_file returned neither a path nor a url"});
    }
    co_return lighter::outcome_error(FetchError{.detail = "segment carries no url, path or file id"});
}

/// Writes `bytes` to `target` atomically unless an identical file exists, a
/// chunk at a time; `bytes` must outlive it.
Task<Result<void>> write_content(std::filesystem::path target, std::string_view bytes) {
    std::error_code ec;
    if (std::filesystem::exists(target, ec)) co_return Result<void>{};
    std::filesystem::create_directories(target.parent_path(), ec);
    if (ec) co_return lighter::outcome_error(Error::storage("cannot create " + target.parent_path().string() + ": " + ec.message()));
    auto temporary = target;
    temporary += ".partial";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        for (usize at = 0; file && at < bytes.size(); at += k_io_chunk) {
            const auto chunk = bytes.substr(at, k_io_chunk);
            file.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            if (at + k_io_chunk < bytes.size()) co_await lighter::yield();
        }
        if (!file) co_return lighter::outcome_error(Error::storage("cannot write " + temporary.string()));
    }
    std::filesystem::rename(temporary, target, ec);
    if (ec) co_return lighter::outcome_error(Error::storage("cannot rename into " + target.string() + ": " + ec.message()));
    co_return Result<void>{};
}

/// One or two sentences about an image, cached by content hash; empty when
/// there is no vision model, the format is unsupported or the call fails.
Task<std::string> describe(Archiver &archiver, const store::ChatKey &chat, const std::string &sha1, const std::string &bytes,
                           const std::string &extension) {
    if (!archiver.vision) co_return std::string();
    const auto media_type = image_media_type(extension);
    if (media_type.empty() || bytes.size() > k_max_vision_bytes) co_return std::string();
    if (auto cached = archiver.store.media_description(sha1); cached.has_value() && !cached->empty()) co_return std::move(*cached);

    provider::History history;
    history.push_back({.role = provider::Role::USER,
                       .parts = {provider::ImagePart{.media_type = std::string(media_type), .data = net::base64_encode(bytes)},
                                 provider::TextPart{.text = std::string(k_vision_prompt)}}});
    store::LlmCall call{.time =
                            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(),
                        .chat = chat,
                        .model = archiver.vision_model};
    // A hung call must not hold an archive slot (and the turns waiting on it).
    const auto items = co_await call_recorded(*archiver.vision, history, {}, k_vision_deadline, archiver.store, call);
    if (items.has_error()) {
        log::warn("describing an image failed: {}", call.error);
        co_return std::string();
    }
    co_return one_line(answer_text(*items));
}

Task<std::string> archive_one(Archiver &archiver, const std::shared_ptr<onebot::Session> &session, const store::StoredMessage &message,
                              MediaRef ref) {
    store::MediaRecord record{
        .chat = message.chat,
        .message_id = message.message_id,
        .segment_index = static_cast<i64>(ref.segment_index),
        .kind = std::string(to_string(ref.kind)),
        .name = ref.name,
        .time = message.time,
    };

    auto fetched = co_await fetch(archiver, session, message.chat, ref);
    if (fetched.has_error()) {
        record.status = fetched.error().too_large ? "too_large" : "failed";
        record.error = fetched.error().detail;
        record.source = is_http(ref.url) ? ref.url : ref.file;
        log::warn("media {} of message {} in {}:{} not archived: {}", ref.segment_index, message.message_id, message.chat.type,
                  message.chat.id, record.error);
    } else {
        const auto &bytes = fetched->bytes;
        record.sha1 = co_await sha1_hex(bytes);
        const auto relative =
            std::filesystem::path("media") / record.sha1.substr(0, 2) / (record.sha1 + detect_extension(bytes, ref.kind, ref.name));
        auto written = co_await write_content(archiver.data_dir / relative, bytes);
        if (written.has_error()) {
            record.status = "failed";
            record.error = written.error().message();
            log::error("{}", record.error);
        } else {
            record.status = "stored";
            record.path = relative.generic_string();
            record.bytes = static_cast<i64>(bytes.size());
        }
        record.source = std::move(fetched->source);
        if (ref.kind == MediaKind::IMAGE || ref.kind == MediaKind::STICKER) {
            record.description = co_await describe(archiver, message.chat, record.sha1, bytes, relative.extension().string());
        }
    }
    if (auto added = archiver.store.add_media(record); added.has_error()) log::error("cannot record media: {}", added.error().message());
    co_return record.description;
}

/// Returns the archive slot even when the task is cancelled mid-download.
struct SlotGuard {
    explicit SlotGuard(lighter::Semaphore &slots) : slots(&slots) {}
    SlotGuard(const SlotGuard &) = delete;
    SlotGuard &operator=(const SlotGuard &) = delete;
    ~SlotGuard() { slots->release(); }

    lighter::Semaphore *slots;
};

/// The nodes of a forward NapCat sent by id only, through get_forward_msg,
/// rendered as a preview; empty when they cannot be had.
Task<std::string> fetch_forward(const std::shared_ptr<onebot::Session> &session, const onebot::Segment &segment) {
    const auto id = onebot::data_string(segment, "id");
    if (!id || id->empty() || !session || !session->is_open()) co_return std::string();
    const auto quoted = glz::write_json(*id).value_or("\"\"");
    auto data = co_await session->call("get_forward_msg", std::format(R"({{"message_id":{0},"id":{0}}})", quoted));
    if (data.has_error()) {
        log::warn("get_forward_msg failed: {}", data.error().message());
        co_return std::string();
    }
    glz::generic_i64 nodes;
    if (glz::read<onebot::k_read_opts>(nodes, *data)) {
        log::warn("get_forward_msg returned undecodable data");
        co_return std::string();
    }
    co_return onebot::render_forward(nodes);
}

std::string extension_of(std::string_view name) {
    const auto dot = name.rfind('.');
    if (dot == std::string_view::npos || dot + 1 >= name.size() || name.size() - dot > 9) return {};
    std::string out = ".";
    for (char c : name.substr(dot + 1)) {
        if (!std::isalnum(static_cast<unsigned char>(c))) return {};
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

} // namespace

std::string_view to_string(MediaKind kind) noexcept {
    switch (kind) {
        case MediaKind::IMAGE: return "image";
        case MediaKind::STICKER: return "sticker";
        case MediaKind::VOICE: return "voice";
        case MediaKind::VIDEO: return "video";
        case MediaKind::FILE: return "file";
    }
    return "unknown";
}

std::vector<MediaRef> collect_media(const onebot::Message &message) {
    std::vector<MediaRef> refs;
    for (usize i = 0; i < message.size(); ++i) {
        const auto &segment = message[i];
        const auto field = [&segment](std::string_view key) { return onebot::data_string(segment, key).value_or(""); };
        MediaRef ref{.segment_index = i, .url = field("url"), .path = field("path"), .file = field("file")};
        if (segment.type == "image") {
            ref.kind = onebot::data_integer(segment, "sub_type").value_or(0) == 1 ? MediaKind::STICKER : MediaKind::IMAGE;
            ref.name = ref.file;
        } else if (segment.type == "mface") {
            ref.kind = MediaKind::STICKER;
            ref.file.clear();
        } else if (segment.type == "record") {
            ref.kind = MediaKind::VOICE;
        } else if (segment.type == "video") {
            ref.kind = MediaKind::VIDEO;
            ref.name = ref.file;
        } else if (segment.type == "file") {
            ref.kind = MediaKind::FILE;
            ref.file_id = field("file_id");
            ref.name = ref.file;
        } else {
            continue;
        }
        refs.push_back(std::move(ref));
    }
    return refs;
}

std::string detect_extension(std::string_view bytes, MediaKind kind, std::string_view name) {
    if (bytes.starts_with("\x89PNG")) return ".png";
    if (bytes.starts_with("\xFF\xD8\xFF")) return ".jpg";
    if (bytes.starts_with("GIF8")) return ".gif";
    if (bytes.starts_with("RIFF") && bytes.size() >= 12 && bytes.substr(8, 4) == "WEBP") return ".webp";
    if (bytes.size() >= 12 && bytes.substr(4, 4) == "ftyp") return ".mp4";
    if (bytes.starts_with("#!SILK") || bytes.starts_with("\x02#!SILK")) return ".silk";
    if (bytes.starts_with("#!AMR")) return ".amr";
    if (bytes.starts_with("ID3")) return ".mp3";
    if (auto from_name = extension_of(name); !from_name.empty()) return from_name;
    switch (kind) {
        case MediaKind::VOICE: return ".audio";
        case MediaKind::VIDEO: return ".video";
        default: return ".bin";
    }
}

std::optional<std::filesystem::path> map_container_path(std::string_view container_path, const ArchiveConfig &config) {
    std::string_view root = config.container_qq_dir;
    while (root.ends_with('/')) root.remove_suffix(1);
    if (root.empty() || !container_path.starts_with(root) || container_path.size() <= root.size() + 1 ||
        container_path[root.size()] != '/') {
        return std::nullopt;
    }
    const std::filesystem::path relative(std::string(container_path.substr(root.size() + 1)));
    for (const auto &part : relative) {
        if (part == "..") return std::nullopt;
    }
    return std::filesystem::path(config.host_qq_dir) / relative;
}

std::string_view image_media_type(std::string_view extension) {
    if (extension == ".png") return "image/png";
    if (extension == ".jpg" || extension == ".jpeg") return "image/jpeg";
    if (extension == ".gif") return "image/gif";
    if (extension == ".webp") return "image/webp";
    return {};
}

Archiver::Archiver(const ArchiveConfig &config, std::filesystem::path data_dir, store::Store &store)
    : config(config), data_dir(std::move(data_dir)), store(store) {}

void Archiver::track(const store::ChatKey &chat, i64 message_id) {
    pending.try_emplace({chat.type, chat.id, message_id}, std::make_shared<lighter::Event>());
}

bool needs_enrichment(const onebot::Message &message, bool media) {
    if (media && !collect_media(message).empty()) return true;
    return std::ranges::any_of(message,
                               [](const onebot::Segment &segment) { return segment.type == "forward" && !onebot::forward_nodes(segment); });
}

Task<> Archiver::archive(std::shared_ptr<onebot::Session> session, store::StoredMessage message, onebot::Message segments,
                         std::string prefix) {
    // Wake waiting turns however this ends.
    struct Done {
        Archiver *archiver;
        std::tuple<std::string, i64, i64> key;
        ~Done() {
            const auto it = archiver->pending.find(key);
            if (it == archiver->pending.end()) return;
            auto event = it->second;
            archiver->pending.erase(it);
            event->set();
        }
    } done{this, {message.chat.type, message.chat.id, message.message_id}};

    std::map<usize, std::string> descriptions;
    for (usize index = 0; index < segments.size(); ++index) {
        if (segments[index].type != "forward" || onebot::forward_nodes(segments[index])) continue;
        if (auto preview = co_await fetch_forward(session, segments[index]); !preview.empty())
            descriptions.emplace(index, std::move(preview));
    }
    const auto refs = config.enabled ? collect_media(segments) : std::vector<MediaRef>{};
    if (!refs.empty()) {
        co_await slots.acquire();
        SlotGuard guard(slots);
        for (const auto &ref : refs) {
            auto description = co_await archive_one(*this, session, message, ref);
            if (!description.empty()) descriptions.emplace(ref.segment_index, std::move(description));
        }
    }
    if (descriptions.empty()) co_return;
    if (auto current = store.message(message.chat, message.message_id);
        current.has_value() && *current && (*current)->text == onebot::k_recalled_text)
        co_return;
    const auto text = prefix + onebot::render_plain(segments, descriptions);
    if (auto updated = store.update_message_text(message.chat, message.message_id, text); updated.has_error()) {
        log::error("cannot update message text: {}", updated.error().message());
    }
}

Task<> Archiver::wait_for(store::ChatKey chat, std::vector<i64> message_ids, std::chrono::milliseconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    for (const auto id : message_ids) {
        const auto it = pending.find({chat.type, chat.id, id});
        if (it == pending.end()) continue;
        auto event = it->second; // keeps the event alive if archive() finishes meanwhile
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        if (left <= std::chrono::milliseconds(0)) co_return;
        std::ignore = co_await lighter::with_timeout(event->wait(), left);
    }
}

} // namespace lilia::archive
