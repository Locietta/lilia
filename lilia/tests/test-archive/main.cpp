#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include <lilia/archive/archiver.h>
#include <lilia/image/preview.h>
#include <lilia/onebot/message.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using archive::MediaKind;
using lilia::test::require;

onebot::Segment segment(std::string type, std::initializer_list<std::pair<const char *, std::string>> fields) {
    onebot::Segment out{.type = std::move(type)};
    out.data = glz::generic_i64::object_t{};
    for (const auto &[key, value] : fields) out.data[key] = value;
    return out;
}

using Rgb = std::array<u8, 3>;

/// A one-frame GIF of palette indices, LZW "uncompressed" (a clear code every
/// 250 pixels keeps codes at 9 bits).
std::string make_gif(int width, int height, const std::vector<u8> &pixels, const std::array<Rgb, 256> &palette) {
    std::string out = "GIF89a";
    const auto u16 = [&out](int value) {
        out.push_back(static_cast<char>(value & 0xff));
        out.push_back(static_cast<char>(value >> 8 & 0xff));
    };
    u16(width);
    u16(height);
    out.append("\xf7\x00\x00", 3);
    for (const auto &colour : palette) out.append(reinterpret_cast<const char *>(colour.data()), 3);
    out.push_back('\x2c');
    u16(0);
    u16(0);
    u16(width);
    u16(height);
    out.push_back('\0');
    out.push_back('\x08');
    std::string data;
    u32 bits = 0;
    int count = 0;
    const auto emit = [&](u32 code) {
        bits |= code << count;
        count += 9;
        while (count >= 8) {
            data.push_back(static_cast<char>(bits & 0xff));
            bits >>= 8;
            count -= 8;
        }
    };
    for (usize i = 0; i < pixels.size(); ++i) {
        if (i % 250 == 0) emit(256);
        emit(pixels[i]);
    }
    emit(257);
    if (count > 0) data.push_back(static_cast<char>(bits & 0xff));
    for (usize at = 0; at < data.size(); at += 255) {
        const auto length = std::min<usize>(255, data.size() - at);
        out.push_back(static_cast<char>(length));
        out.append(data, at, length);
    }
    out.push_back('\0');
    out.push_back('\x3b');
    return out;
}

void test_preview() {
    std::array<Rgb, 256> palette{};
    for (int i = 0; i < 256; ++i) palette[static_cast<usize>(i)] = {static_cast<u8>(i), static_cast<u8>(255 - i), static_cast<u8>(i * 7)};
    const auto small = make_gif(32, 16, std::vector<u8>(32 * 16, 3), palette);
    const auto kept = image::for_model(small);
    require(kept && kept->media_type == "image/gif" && kept->bytes == small, "a small picture goes as it is");
    const auto wide = make_gif(2400, 30, std::vector<u8>(2400 * 30, 7), palette);
    const auto scaled = image::for_model(wide);
    require(scaled && scaled->media_type == "image/jpeg" && scaled->bytes.starts_with("\xFF\xD8\xFF") && scaled->bytes.size() < wide.size(),
            "a large one is scaled down to a JPEG");
    require(image::for_model(wide, 4096, 1'000'000) && image::for_model(wide, 4096, 1'000'000)->media_type == "image/gif",
            "within both limits it stays");
    const std::string webp = std::string("RIFF\x10\x00\x00\x00WEBPVP8 ", 16) + std::string(20, 'x');
    require(image::for_model(webp) && image::for_model(webp)->media_type == "image/webp", "a small WebP goes as it is");
    require(!image::for_model("not a picture") && !image::for_model("GIF89a") && !image::for_model(""), "nothing else");
    for (usize length = 0; length < wide.size(); length += 997) std::ignore = image::for_model(wide.substr(0, length));
}

void test_collect() {
    auto sticker = segment("image", {{"file", "a.gif"}, {"url", "https://example.invalid/a.gif"}});
    sticker.data["sub_type"] = int64_t{1};
    const onebot::Message message{
        onebot::text_segment("看"),
        segment("image", {{"file", "b.jpg"}, {"url", "https://example.invalid/b.jpg"}}),
        sticker,
        segment("mface", {{"url", "https://gxh.vip.qq.com/raw300.gif"}, {"summary", "[开心]"}}),
        segment("record", {{"file", "v.amr"}, {"path", "/app/.config/QQ/x/v.amr"}}),
        segment("file", {{"file", "doc.pdf"}, {"file_id", "/abc"}}),
        onebot::at_segment(1),
    };
    const auto refs = archive::collect_media(message);
    require(refs.size() == 5, "text and at are not media");
    require(refs[0].segment_index == 1 && refs[0].kind == MediaKind::IMAGE && refs[0].url.ends_with("b.jpg"), "image");
    require(refs[1].kind == MediaKind::STICKER, "image sub_type 1 is a sticker");
    require(refs[2].kind == MediaKind::STICKER && refs[2].file.empty(), "mface is a sticker fetched by url");
    require(refs[3].kind == MediaKind::VOICE && refs[3].path == "/app/.config/QQ/x/v.amr", "voice keeps its container path");
    require(refs[4].kind == MediaKind::FILE && refs[4].file_id == "/abc" && refs[4].name == "doc.pdf", "group file");
}

void test_extensions() {
    require(archive::detect_extension("\x89PNG\r\n", MediaKind::IMAGE, "x.jpg") == ".png", "magic bytes win over the name");
    require(archive::detect_extension("\xFF\xD8\xFF\xE0", MediaKind::IMAGE, "") == ".jpg", "jpeg");
    require(archive::detect_extension("GIF89a", MediaKind::STICKER, "") == ".gif", "gif");
    require(archive::detect_extension(std::string("RIFF\0\0\0\0WEBPVP8", 15), MediaKind::IMAGE, "") == ".webp", "webp");
    require(archive::detect_extension(std::string("\0\0\0\x18"
                                                  "ftypmp42",
                                                  12),
                                      MediaKind::VIDEO, "") == ".mp4",
            "mp4");
    require(archive::detect_extension("\x02#!SILK_V3", MediaKind::VOICE, "") == ".silk", "silk voice");
    require(archive::detect_extension("plain", MediaKind::FILE, "Report.PDF") == ".pdf", "name extension, lower-cased");
    require(archive::detect_extension("plain", MediaKind::FILE, "evil.p/df") == ".bin", "odd extensions fall back");
    require(archive::detect_extension("plain", MediaKind::VOICE, "") == ".audio", "per-kind default");
}

void test_paths() {
    ArchiveConfig config{.container_qq_dir = "/app/.config/QQ/", .host_qq_dir = "/home/loia/services/napcat/qq"};
    const auto mapped = archive::map_container_path("/app/.config/QQ/nt_qq_1/nt_data/Ptt/a.amr", config);
    require(mapped && mapped->generic_string() == "/home/loia/services/napcat/qq/nt_qq_1/nt_data/Ptt/a.amr", "path is mapped");
    require(!archive::map_container_path("/app/napcat/temp/a.mp3", config), "paths outside the mount are refused");
    require(!archive::map_container_path("/app/.config/QQ2/a", config), "a sibling prefix is not the mount");
    require(!archive::map_container_path("/app/.config/QQ/../../etc/passwd", config), "parent traversal is refused");
}

void test_descriptions() {
    require(archive::image_media_type(".png") == "image/png" && archive::image_media_type(".jpg") == "image/jpeg" &&
                archive::image_media_type(".gif") == "image/gif" && archive::image_media_type(".webp") == "image/webp",
            "image media types");
    require(archive::image_media_type(".amr").empty() && archive::image_media_type(".mp4").empty(), "non-images have none");

    auto sticker = segment("image", {{"file", "a.gif"}});
    sticker.data["sub_type"] = int64_t{1};
    const onebot::Message message{onebot::text_segment("看"), segment("image", {{"file", "b.jpg"}}), sticker,
                                  segment("mface", {{"summary", "[开心]"}}), segment("record", {})};
    require(onebot::render_plain(message, {{1, "一只猫"}, {2, "狗头"}, {3, "开心地转圈"}, {4, "不会用到"}}) ==
                "看[图片: 一只猫][表情: 狗头][表情: 开心地转圈][语音]",
            "described images and stickers render their descriptions; other segments ignore them");
    require(onebot::render_plain(message) == "看[图片][表情][表情: 开心][语音]", "without descriptions nothing changes");

    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    const store::ChatKey chat{.type = "group", .id = 7};
    require(db->add_media({.chat = chat, .message_id = 1, .kind = "image", .status = "stored", .sha1 = "abc", .description = "一只猫"})
                .has_value(),
            "media with a description");
    require(*db->media_description("abc") == "一只猫", "description cache hit by content");
    require(db->media_description("zzz")->empty(), "cache miss");
    require(db->add_message({.chat = chat, .message_id = 1, .text = "[图片]"}).has_value(), "message");
    require(db->update_message_text(chat, 1, "[图片: 一只猫]").has_value(), "update");
    require(db->recent_messages(chat, 1)->front().text == "[图片: 一只猫]", "the message text is replaced");
}

void test_media_rows() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    const store::ChatKey chat{.type = "group", .id = 7};
    require(
        db->add_media({.chat = chat, .message_id = 5, .segment_index = 2, .kind = "voice", .status = "failed", .error = "x"}).has_value(),
        "failed row");
    require(db->add_media({.chat = chat,
                           .message_id = 5,
                           .segment_index = 2,
                           .kind = "voice",
                           .status = "stored",
                           .sha1 = "ab",
                           .path = "media/ab/ab.amr",
                           .bytes = 3})
                .has_value(),
            "a retry replaces the failed row");
    require(db->add_media({.chat = chat, .message_id = 5, .segment_index = 1, .kind = "image", .status = "too_large"}).has_value(), "row");
    auto rows = db->message_media(chat, 5);
    require(rows.has_value() && rows->size() == 2, "one row per segment");
    require((*rows)[0].segment_index == 1 && (*rows)[1].status == "stored" && (*rows)[1].path == "media/ab/ab.amr", "rows in order");
}

} // namespace

int main() {
    try {
        test_collect();
        test_preview();
        test_extensions();
        test_paths();
        test_media_rows();
        test_descriptions();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
