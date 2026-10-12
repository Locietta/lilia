#include "frames.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>

// Static, so a test can link its own stb with more formats; GIF only, since
// these files come from strangers.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_GIF
#define STBI_NO_STDIO
#include <stb_image.h>
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO
#include <stb_image_write.h>

namespace lilia::image {

namespace {

constexpr int k_thumbnail = 64;
constexpr u8 k_gutter_grey = 200;

struct Rgb {
    int width = 0;
    int height = 0;
    std::vector<u8> pixels;
};

/// One RGBA frame over white.
Rgb over_white(const u8 *rgba, int width, int height) {
    Rgb out{.width = width, .height = height};
    const usize count = static_cast<usize>(width) * static_cast<usize>(height);
    out.pixels.resize(count * 3);
    for (usize i = 0; i < count; ++i) {
        const u32 alpha = rgba[i * 4 + 3];
        for (usize c = 0; c < 3; ++c) {
            out.pixels[i * 3 + c] = static_cast<u8>((rgba[i * 4 + c] * alpha + 255 * (255 - alpha) + 127) / 255);
        }
    }
    return out;
}

Rgb resized(const Rgb &image, int width, int height) {
    Rgb out{.width = width, .height = height};
    out.pixels.resize(static_cast<usize>(width) * static_cast<usize>(height) * 3);
    stbir_resize_uint8_srgb(image.pixels.data(), image.width, image.height, image.width * 3, out.pixels.data(), width, height, width * 3,
                            STBIR_RGB);
    return out;
}

double mean_squared_difference(const Rgb &a, const Rgb &b) {
    double sum = 0;
    for (usize i = 0; i < a.pixels.size(); ++i) {
        const double d = static_cast<double>(a.pixels[i]) - static_cast<double>(b.pixels[i]);
        sum += d * d;
    }
    return a.pixels.empty() ? 0 : sum / static_cast<double>(a.pixels.size());
}

/// `count` of `candidates` (frame indices in order), spread evenly over the
/// time between the first and the last, which are always kept.
std::vector<usize> spread(const std::vector<usize> &candidates, const std::vector<i64> &starts, usize count) {
    if (candidates.size() <= count) return candidates;
    const double first = static_cast<double>(starts[candidates.front()]);
    const double last = static_cast<double>(starts[candidates.back()]);
    std::vector<usize> out;
    usize low = 0;
    for (usize k = 0; k < count; ++k) {
        const double target = first + (last - first) * static_cast<double>(k) / static_cast<double>(count - 1);
        // Leave enough candidates for the picks still to come.
        const usize high = candidates.size() - (count - k);
        usize best = low;
        for (usize i = low; i <= high; ++i) {
            if (std::abs(static_cast<double>(starts[candidates[i]]) - target) <
                std::abs(static_cast<double>(starts[candidates[best]]) - target)) {
                best = i;
            }
        }
        out.push_back(candidates[best]);
        low = best + 1;
    }
    return out;
}

} // namespace

std::optional<GifShape> gif_shape(std::string_view bytes) {
    if (bytes.size() < 13 || !(bytes.starts_with("GIF87a") || bytes.starts_with("GIF89a"))) return std::nullopt;
    const auto byte = [&](usize at) { return static_cast<u8>(bytes[at]); };
    GifShape shape{.width = static_cast<u32>(byte(6) | byte(7) << 8), .height = static_cast<u32>(byte(8) | byte(9) << 8)};
    usize at = 13;
    if (byte(10) & 0x80) at += usize{3} << ((byte(10) & 7) + 1);
    // Data sub-blocks: a length byte, that many bytes, until a zero length.
    const auto skip_blocks = [&](usize from) -> std::optional<usize> {
        while (from < bytes.size()) {
            const usize length = byte(from++);
            if (length == 0) return from;
            from += length;
        }
        return std::nullopt;
    };
    while (at < bytes.size()) {
        const u8 tag = byte(at++);
        if (tag == 0x3b) break; // trailer
        if (tag == 0x21) {      // extension: a label, then sub-blocks
            auto next = at < bytes.size() ? skip_blocks(at + 1) : std::nullopt;
            if (!next) return std::nullopt;
            at = *next;
        } else if (tag == 0x2c) { // image descriptor
            if (at + 9 > bytes.size()) return std::nullopt;
            const u8 packed = byte(at + 8);
            at += 9;
            if (packed & 0x80) at += usize{3} << ((packed & 7) + 1);
            // The LZW minimum code size, then the image data.
            auto next = at < bytes.size() ? skip_blocks(at + 1) : std::nullopt;
            if (!next) return std::nullopt;
            at = *next;
            ++shape.frames;
        } else {
            break; // trailing garbage ends the file, as for decoders
        }
    }
    if (shape.frames == 0) return std::nullopt;
    return shape;
}

std::optional<FrameGrid> frame_grid(std::string_view bytes, const GridOptions &options) {
    const auto shape = gif_shape(bytes);
    if (!shape || shape->frames < 2 || options.max_frames < 2) return std::nullopt;
    if (shape->width == 0 || shape->height == 0 || shape->width > options.max_side || shape->height > options.max_side) return std::nullopt;
    if (u64{shape->width} * shape->height * 4 * shape->frames > options.max_decoded_bytes) return std::nullopt;

    int *delays = nullptr;
    int width = 0;
    int height = 0;
    int count = 0;
    int channels = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(stbi_load_gif_from_memory(reinterpret_cast<const stbi_uc *>(bytes.data()),
                                                                                          static_cast<int>(bytes.size()), &delays, &width,
                                                                                          &height, &count, &channels, 4),
                                                                &stbi_image_free);
    std::unique_ptr<int, decltype(&std::free)> delays_owner(delays, &std::free);
    if (!pixels || count < 2 || width <= 0 || height <= 0) return std::nullopt;

    // Browsers show delays of 10 ms or less as 100 ms.
    std::vector<i64> starts;
    i64 duration = 0;
    for (int i = 0; i < count; ++i) {
        starts.push_back(duration);
        duration += delays && delays[i] > 10 ? delays[i] : 100;
    }
    const usize frame_bytes = static_cast<usize>(width) * static_cast<usize>(height) * 4;
    const auto frame = [&](usize index) { return over_white(pixels.get() + index * frame_bytes, width, height); };

    // Frames that differ from the last kept one.
    std::vector<usize> distinct{0};
    Rgb last = resized(frame(0), k_thumbnail, k_thumbnail);
    for (usize i = 1; i < static_cast<usize>(count); ++i) {
        auto thumbnail = resized(frame(i), k_thumbnail, k_thumbnail);
        if (mean_squared_difference(thumbnail, last) > options.min_mse) {
            distinct.push_back(i);
            last = std::move(thumbnail);
        }
    }
    // Small motions (a shake of the head) stay under the threshold: then
    // spread over all frames instead.
    if (distinct.size() < 3 && count >= 6) {
        distinct.clear();
        for (usize i = 0; i < static_cast<usize>(count); ++i) distinct.push_back(i);
    }
    if (distinct.size() < 2) return std::nullopt;
    const auto chosen = spread(distinct, starts, options.max_frames);

    const double scale = static_cast<double>(options.cell) / std::max(width, height);
    const int cell_width = std::max(1, static_cast<int>(std::lround(width * scale)));
    const int cell_height = std::max(1, static_cast<int>(std::lround(height * scale)));
    const int columns = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(chosen.size()))));
    const int rows = (static_cast<int>(chosen.size()) + columns - 1) / columns;
    const int gutter = static_cast<int>(options.gutter);
    Rgb grid{.width = columns * cell_width + (columns - 1) * gutter, .height = rows * cell_height + (rows - 1) * gutter};
    grid.pixels.assign(static_cast<usize>(grid.width) * static_cast<usize>(grid.height) * 3, k_gutter_grey);
    // Empty cells after the last frame are white, not grey.
    for (int cell = 0; cell < columns * rows; ++cell) {
        const int x0 = (cell % columns) * (cell_width + gutter);
        const int y0 = (cell / columns) * (cell_height + gutter);
        const auto tile =
            static_cast<usize>(cell) < chosen.size() ?
                resized(frame(chosen[static_cast<usize>(cell)]), cell_width, cell_height) :
                Rgb{cell_width, cell_height, std::vector<u8>(static_cast<usize>(cell_width) * static_cast<usize>(cell_height) * 3, 255)};
        for (int y = 0; y < cell_height; ++y) {
            std::copy_n(tile.pixels.begin() + static_cast<std::ptrdiff_t>(y) * cell_width * 3, cell_width * 3,
                        grid.pixels.begin() + (static_cast<std::ptrdiff_t>(y0 + y) * grid.width + x0) * 3);
        }
    }

    FrameGrid out{.indices = chosen, .frames = chosen.size(), .total = static_cast<usize>(count), .duration_ms = duration};
    const auto append = [](void *context, void *data, int size) {
        static_cast<std::string *>(context)->append(static_cast<const char *>(data), static_cast<usize>(size));
    };
    if (!stbi_write_jpg_to_func(append, &out.jpeg, grid.width, grid.height, 3, grid.pixels.data(), options.quality)) return std::nullopt;
    return out;
}

} // namespace lilia::image
