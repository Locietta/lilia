#include "preview.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

// Static, so nothing else that links stb can clash; only the formats QQ
// pictures come in, since these files come from strangers.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
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

/// Larger pictures are not decoded at all.
constexpr u64 k_max_pixels = u64{40} << 20;

std::string_view media_type_of(std::string_view bytes) {
    if (bytes.starts_with("\x89PNG")) return "image/png";
    if (bytes.starts_with("\xFF\xD8\xFF")) return "image/jpeg";
    if (bytes.starts_with("GIF8")) return "image/gif";
    if (bytes.starts_with("RIFF") && bytes.size() >= 12 && bytes.substr(8, 4) == "WEBP") return "image/webp";
    return {};
}

} // namespace

std::optional<Picture> for_model(std::string_view bytes, u32 max_side, usize max_bytes) {
    const auto media_type = media_type_of(bytes);
    if (media_type.empty()) return std::nullopt;
    if (media_type == "image/webp") {
        if (bytes.size() > max_bytes) return std::nullopt;
        return Picture{.media_type = std::string(media_type), .bytes = std::string(bytes)};
    }
    const auto *data = reinterpret_cast<const stbi_uc *>(bytes.data());
    const int size = static_cast<int>(std::min<usize>(bytes.size(), 0x7fffffff));
    int width = 0;
    int height = 0;
    int channels = 0;
    if (!stbi_info_from_memory(data, size, &width, &height, &channels) || width <= 0 || height <= 0) return std::nullopt;
    if (static_cast<u32>(std::max(width, height)) <= max_side && bytes.size() <= max_bytes) {
        return Picture{.media_type = std::string(media_type), .bytes = std::string(bytes)};
    }
    if (u64(width) * u64(height) > k_max_pixels) return std::nullopt;

    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> rgba(stbi_load_from_memory(data, size, &width, &height, &channels, 4),
                                                              &stbi_image_free);
    if (!rgba) return std::nullopt;
    const usize pixels = static_cast<usize>(width) * static_cast<usize>(height);
    std::vector<u8> rgb(pixels * 3);
    for (usize i = 0; i < pixels; ++i) {
        const u32 alpha = rgba.get()[i * 4 + 3];
        for (usize c = 0; c < 3; ++c) rgb[i * 3 + c] = static_cast<u8>((rgba.get()[i * 4 + c] * alpha + 255 * (255 - alpha) + 127) / 255);
    }
    const double scale = std::min(1.0, static_cast<double>(max_side) / std::max(width, height));
    const int out_width = std::max(1, static_cast<int>(std::lround(width * scale)));
    const int out_height = std::max(1, static_cast<int>(std::lround(height * scale)));
    std::vector<u8> scaled(static_cast<usize>(out_width) * static_cast<usize>(out_height) * 3);
    stbir_resize_uint8_srgb(rgb.data(), width, height, width * 3, scaled.data(), out_width, out_height, out_width * 3, STBIR_RGB);

    Picture out{.media_type = "image/jpeg"};
    const auto append = [](void *context, void *chunk, int length) {
        static_cast<std::string *>(context)->append(static_cast<const char *>(chunk), static_cast<usize>(length));
    };
    if (!stbi_write_jpg_to_func(append, &out.bytes, out_width, out_height, 3, scaled.data(), 85)) return std::nullopt;
    return out;
}

} // namespace lilia::image
