#include "archiver.h"

#include <array>
#include <cctype>
#include <chrono>
#include <format>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

#include <curl/curl.h>
#include <glaze/json.hpp>

#include <lighter/http/bound_client.h>

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

std::string sha1_hex(std::string_view bytes) {
    static constexpr std::string_view k_hex = "0123456789abcdef";
    std::string out;
    for (const u8 byte : net::sha1(bytes)) {
        out.push_back(k_hex[byte >> 4]);
        out.push_back(k_hex[byte & 0xf]);
    }
    return out;
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

FetchResult read_container_file(const Archiver &archiver, std::string_view container_path) {
    const auto host = map_container_path(container_path, archiver.config);
    if (!host) return lighter::outcome_error(FetchError{.detail = "path outside the shared QQ directory: " + std::string(container_path)});
    std::error_code ec;
    const auto size = std::filesystem::file_size(*host, ec);
    if (ec) return lighter::outcome_error(FetchError{.detail = "cannot stat " + host->string() + ": " + ec.message()});
    if (size > archiver.config.max_file_bytes) return too_large(size, archiver.config.max_file_bytes);
    std::ifstream file(*host, std::ios::binary);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    if (!file && !file.eof()) return lighter::outcome_error(FetchError{.detail = "cannot read " + host->string()});
    return Fetched{.bytes = buffer.str(), .source = "napcat:" + std::string(container_path)};
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
    if (!ref.path.empty()) co_return read_container_file(archiver, ref.path);
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
        if (info->file && info->file->starts_with('/')) co_return read_container_file(archiver, *info->file);
        if (info->url && is_http(*info->url)) co_return co_await download(archiver, *info->url);
        co_return lighter::outcome_error(FetchError{.detail = "get_file returned neither a path nor a url"});
    }
    co_return lighter::outcome_error(FetchError{.detail = "segment carries no url, path or file id"});
}

/// Writes `bytes` to `target` atomically unless an identical file exists.
Result<void> write_content(const std::filesystem::path &target, std::string_view bytes) {
    std::error_code ec;
    if (std::filesystem::exists(target, ec)) return {};
    std::filesystem::create_directories(target.parent_path(), ec);
    if (ec) return lighter::outcome_error(Error::storage("cannot create " + target.parent_path().string() + ": " + ec.message()));
    auto temporary = target;
    temporary += ".partial";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!file) return lighter::outcome_error(Error::storage("cannot write " + temporary.string()));
    }
    std::filesystem::rename(temporary, target, ec);
    if (ec) return lighter::outcome_error(Error::storage("cannot rename into " + target.string() + ": " + ec.message()));
    return {};
}

Task<> archive_one(Archiver &archiver, const std::shared_ptr<onebot::Session> &session, const store::StoredMessage &message, MediaRef ref) {
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
        record.sha1 = sha1_hex(bytes);
        const auto relative =
            std::filesystem::path("media") / record.sha1.substr(0, 2) / (record.sha1 + detect_extension(bytes, ref.kind, ref.name));
        auto written = write_content(archiver.data_dir / relative, bytes);
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
    }
    if (auto added = archiver.store.add_media(record); added.has_error()) log::error("cannot record media: {}", added.error().message());
}

/// Returns the archive slot even when the task is cancelled mid-download.
struct SlotGuard {
    explicit SlotGuard(lighter::Semaphore &slots) : slots(&slots) {}
    SlotGuard(const SlotGuard &) = delete;
    SlotGuard &operator=(const SlotGuard &) = delete;
    ~SlotGuard() { slots->release(); }

    lighter::Semaphore *slots;
};

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

Archiver::Archiver(const ArchiveConfig &config, std::filesystem::path data_dir, store::Store &store)
    : config(config), data_dir(std::move(data_dir)), store(store) {}

Task<> Archiver::archive(std::shared_ptr<onebot::Session> session, store::StoredMessage message, onebot::Message segments) {
    const auto refs = collect_media(segments);
    if (refs.empty()) co_return;
    co_await slots.acquire();
    SlotGuard guard(slots);
    for (const auto &ref : refs) co_await archive_one(*this, session, message, ref);
}

} // namespace lilia::archive
