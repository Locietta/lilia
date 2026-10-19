#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <lilia/chat/session.h>
#include <lilia/provider/history.h>
#include <lilia/store/store.h>

namespace lilia::chat {

// Models see the pictures of a chat themselves (2026-10-19): a photo is not
// described in advance (one sentence about a picture seen once says little),
// it goes to the planner as an image beside its line. Stickers the collection
// has learned read as their description instead; one it has not learned yet
// goes as a picture too.

/// Pictures read from the media archive, by content hash; nullopt for a file
/// that cannot be shown (missing, not a picture, too large).
struct PictureCache {
    std::unordered_map<std::string, std::optional<provider::ImagePart>> by_sha1;
};

/// The pictures of `rows` models should be shown, by row: the stored photos,
/// and stickers stored without a description (hers have none: what she
/// sends is not archived, but a replay may give her lines with pictures). Read
/// from the archive under `data_dir`, sized by image::for_model.
Pictures pictures_of(store::Store &store, const std::filesystem::path &data_dir, const std::vector<store::StoredMessage> &rows,
                     PictureCache &cache);

} // namespace lilia::chat
