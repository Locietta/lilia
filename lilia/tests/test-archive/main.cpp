#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include <lilia/archive/archiver.h>
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

/// A database created by schema 3 gains media.description in place.
void test_migration() {
    const auto dir = std::filesystem::temp_directory_path() / ("lilia-test-migrate-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const auto path = dir / "lilia.db";
    {
        auto old = store::Database::open(path);
        require(old.has_value(), "open old");
        require(old->execute("CREATE TABLE media (id INTEGER PRIMARY KEY, chat_type TEXT NOT NULL, chat_id INTEGER NOT NULL, "
                             "message_id INTEGER NOT NULL, segment_index INTEGER NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, "
                             "sha1 TEXT NOT NULL, path TEXT NOT NULL, bytes INTEGER NOT NULL, name TEXT NOT NULL, source TEXT NOT NULL, "
                             "error TEXT NOT NULL, time INTEGER NOT NULL, UNIQUE (chat_type, chat_id, message_id, segment_index)); "
                             "INSERT INTO media VALUES (1, 'group', 7, 1, 0, 'image', 'stored', 'old', 'p', 1, '', '', '', 1); "
                             "PRAGMA user_version = 3;")
                    .has_value(),
                "old schema");
    }
    {
        auto migrated = store::Store::open(path);
        require(migrated.has_value(), migrated ? std::string() : migrated.error().message());
        auto rows = migrated->message_media({.type = "group", .id = 7}, 1);
        require(rows.has_value() && rows->size() == 1 && rows->front().sha1 == "old" && rows->front().description.empty(),
                "old rows survive with an empty description");
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
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
        test_extensions();
        test_paths();
        test_media_rows();
        test_descriptions();
        test_migration();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
