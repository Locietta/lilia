#pragma once

#include <optional>
#include <string>
#include <string_view>

#include <lighter/types.hpp>

namespace lilia::image {

using namespace lighter::types;

// Pictures for vision models. A model reads a picture as at most ~1.5k pixels
// a side anyway, and the planner's window sends its pictures with every
// round, so large ones go smaller.

struct Picture {
    std::string media_type; ///< "image/png", "image/jpeg", ...
    std::string bytes;
};

/// `bytes` as they are when small (each side at most `max_side`, at most
/// `max_bytes`); otherwise decoded (PNG, JPEG, a GIF's first frame) over
/// white, scaled to fit `max_side` and encoded as JPEG. A small WebP goes as
/// it is (it cannot be decoded here). nullopt when it is no picture, or too
/// large to decode.
std::optional<Picture> for_model(std::string_view bytes, u32 max_side = 1024, usize max_bytes = 400 * 1024);

} // namespace lilia::image
