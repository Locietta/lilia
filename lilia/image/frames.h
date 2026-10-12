#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/types.hpp>

namespace lilia::image {

using namespace lighter::types;

// Animated stickers for vision models, which see only the first frame of a
// GIF: a few distinct frames in time order, tiled into one picture (after
// MaiBot's frame strip, as a grid so it stays under the models' downscale
// limit).

struct GridOptions {
    /// Frames in the grid at most (3 x 3).
    usize max_frames = 9;
    /// Longest side of one frame in the grid.
    u32 cell = 256;
    /// Grey lines between frames, so neighbours do not read as one picture.
    u32 gutter = 4;
    /// A frame counts as new when its mean squared difference from the last
    /// kept frame (on 64 x 64 thumbnails, 0-255 per channel) is above this.
    double min_mse = 1000;
    int quality = 85;
    /// GIFs larger than this are not decoded (they come from strangers).
    u32 max_side = 2048;
    u64 max_decoded_bytes = u64{64} << 20;
};

struct FrameGrid {
    std::string jpeg;
    /// Frames shown (their indices in the file, in order), and in the file.
    std::vector<usize> indices;
    usize frames = 0;
    usize total = 0;
    i64 duration_ms = 0;
};

struct GifShape {
    u32 width = 0;
    u32 height = 0;
    usize frames = 0;
};

/// Size and frame count from a GIF's block structure, without decoding;
/// nullopt when `bytes` is not a well-formed GIF.
std::optional<GifShape> gif_shape(std::string_view bytes);

/// An animated GIF as one JPEG of up to max_frames distinct frames over white,
/// left to right and top to bottom in time order. nullopt for a still image,
/// another format, or anything that cannot be decoded within the limits.
std::optional<FrameGrid> frame_grid(std::string_view bytes, const GridOptions &options = {});

} // namespace lilia::image
