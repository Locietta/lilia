#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/async/runtime/sync.h>
#include <lighter/async/runtime/task.h>
#include <lighter/http/client.h>

#include <lilia/config.h>
#include <lilia/onebot/session.h>
#include <lilia/store/store.h>

namespace lilia::archive {

// Keeps the media of allowed chats permanently. The QQ client's own cache is
// pruned after a few days (deployment: napcat-cleanup.timer) and Tencent's
// download links expire, so lilia copies pictures, stickers, voice, video and
// files into <data_dir>/media as soon as a message arrives.

enum struct MediaKind {
    IMAGE,
    STICKER,
    VOICE,
    VIDEO,
    FILE,
};

std::string_view to_string(MediaKind kind) noexcept;

/// Where one media segment can be fetched from, as far as the message says.
struct MediaRef {
    usize segment_index = 0;
    MediaKind kind = MediaKind::IMAGE;
    std::string url;     ///< direct http(s) download, when present
    std::string path;    ///< absolute path inside the NapCat container, when present
    std::string file;    ///< NapCat file identifier for get_file
    std::string file_id; ///< group file id for get_group_file_url
    std::string name;    ///< original file name, when known
};

std::vector<MediaRef> collect_media(const onebot::Message &message);

/// Extension (with the dot) from magic bytes, falling back to the original
/// name and then to a per-kind default.
std::string detect_extension(std::string_view bytes, MediaKind kind, std::string_view name);

/// Maps a path inside the NapCat container to the bind-mounted host path.
/// nullopt when the path is outside the shared QQ data directory.
std::optional<std::filesystem::path> map_container_path(std::string_view container_path, const ArchiveConfig &config);

struct Archiver {
    Archiver(const ArchiveConfig &config, std::filesystem::path data_dir, store::Store &store);

    /// Archives every media segment of one stored message. Never fails: each
    /// outcome, including errors, becomes a row in the media table.
    lighter::Task<> archive(std::shared_ptr<onebot::Session> session, store::StoredMessage message, onebot::Message segments);

    const ArchiveConfig &config;
    std::filesystem::path data_dir;
    store::Store &store;
    lighter::http::Client http;
    lighter::Semaphore slots{2};
};

} // namespace lilia::archive
