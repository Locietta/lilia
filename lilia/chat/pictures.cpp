#include "pictures.h"

#include <fstream>
#include <sstream>

#include <lilia/image/preview.h>
#include <lilia/net/codec.h>

namespace lilia::chat {

namespace {

/// Pictures kept in memory at most; past it the cache starts over.
constexpr usize k_cached_pictures = 256;

std::optional<provider::ImagePart> load(const std::filesystem::path &file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return std::nullopt;
    std::ostringstream bytes;
    bytes << in.rdbuf();
    const auto picture = image::for_model(bytes.str());
    if (!picture) return std::nullopt;
    return provider::ImagePart{.media_type = picture->media_type, .data = net::base64_encode(picture->bytes)};
}

} // namespace

Pictures pictures_of(store::Store &store, const std::filesystem::path &data_dir, const std::vector<store::StoredMessage> &rows,
                     PictureCache &cache) {
    Pictures out;
    for (const auto &row : rows) {
        if (row.from_self) continue;
        auto media = store.message_media(row.chat, row.message_id);
        if (!media.has_value()) continue;
        for (const auto &record : *media) {
            if (record.status != "stored" || record.sha1.empty()) continue;
            if (record.kind != "image" && !(record.kind == "sticker" && record.description.empty())) continue;
            auto found = cache.by_sha1.find(record.sha1);
            if (found == cache.by_sha1.end()) {
                if (cache.by_sha1.size() >= k_cached_pictures) cache.by_sha1.clear();
                found = cache.by_sha1.emplace(record.sha1, load(data_dir / record.path)).first;
            }
            if (found->second) out[row.row_id].push_back(*found->second);
        }
    }
    return out;
}

} // namespace lilia::chat
