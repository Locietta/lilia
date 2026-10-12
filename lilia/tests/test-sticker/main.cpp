#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <lighter/async/io/loop.h>

#include <lilia/image/frames.h>
#include <lilia/onebot/message.h>
#include <lilia/sticker/sticker.h>
#include <lilia/store/store.h>
#include <lilia/tests/support/fake_provider.h>
#include <lilia/tests/support/fake_ws_peer.h>

// The test's own stb, with JPEG, to read the grids back.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include <stb_image.h>

namespace {

using namespace lilia;
using lilia::test::require;

using Rgb = std::array<u8, 3>;

/// A GIF with a 256-colour palette whose frames are palette indices, LZW
/// "uncompressed" (a clear code every 250 pixels keeps codes at 9 bits).
std::string make_gif(int width, int height, const std::vector<std::vector<u8>> &frames, const std::array<Rgb, 256> &palette,
                     int delay_cs = 10, int transparent = -1) {
    std::string out = "GIF89a";
    const auto u16 = [&out](int value) {
        out.push_back(static_cast<char>(value & 0xff));
        out.push_back(static_cast<char>(value >> 8 & 0xff));
    };
    u16(width);
    u16(height);
    out.append("\xf7\x00\x00", 3);
    for (const auto &colour : palette) out.append(reinterpret_cast<const char *>(colour.data()), 3);
    for (const auto &frame : frames) {
        out += "\x21\xf9\x04";
        out.push_back(static_cast<char>(1 << 2 | (transparent >= 0 ? 1 : 0)));
        u16(delay_cs);
        out.push_back(static_cast<char>(transparent >= 0 ? transparent : 0));
        out.push_back('\0');
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
        for (usize i = 0; i < frame.size(); ++i) {
            if (i % 250 == 0) emit(256);
            emit(frame[i]);
        }
        emit(257);
        if (count > 0) data.push_back(static_cast<char>(bits & 0xff));
        for (usize at = 0; at < data.size(); at += 255) {
            const auto length = std::min<usize>(255, data.size() - at);
            out.push_back(static_cast<char>(length));
            out.append(data, at, length);
        }
        out.push_back('\0');
    }
    out.push_back('\x3b');
    return out;
}

std::array<Rgb, 256> test_palette() {
    std::array<Rgb, 256> palette{};
    for (int i = 0; i < 256; ++i) palette[static_cast<usize>(i)] = {static_cast<u8>(i), static_cast<u8>(255 - i), static_cast<u8>(i * 7)};
    palette[0] = {255, 255, 255};
    palette[1] = {0, 0, 0};
    return palette;
}

/// A white 32 x 32 frame with a black 8 x 8 square at (x, y).
std::vector<u8> square_at(int x, int y) {
    std::vector<u8> frame(32 * 32, 0);
    for (int row = y; row < y + 8; ++row) {
        for (int column = x; column < x + 8; ++column) frame[static_cast<usize>(row * 32 + column)] = 1;
    }
    return frame;
}

struct Decoded {
    int width = 0;
    int height = 0;
    std::vector<u8> pixels;
    Rgb at(int x, int y) const {
        const auto i = static_cast<usize>((y * width + x) * 3);
        return {pixels[i], pixels[i + 1], pixels[i + 2]};
    }
};

Decoded decode_jpeg(const std::string &jpeg) {
    Decoded out;
    int channels = 0;
    auto *pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc *>(jpeg.data()), static_cast<int>(jpeg.size()), &out.width,
                                         &out.height, &channels, 3);
    require(pixels != nullptr, "the grid is a JPEG");
    out.pixels.assign(pixels, pixels + static_cast<usize>(out.width * out.height * 3));
    stbi_image_free(pixels);
    return out;
}

bool looks_like(Rgb colour, int r, int g, int b) {
    return std::abs(colour[0] - r) < 24 && std::abs(colour[1] - g) < 24 && std::abs(colour[2] - b) < 24;
}

void test_frames() {
    const auto palette = test_palette();
    const auto still = make_gif(32, 32, {square_at(0, 0)}, palette);
    require(image::gif_shape(still) && image::gif_shape(still)->frames == 1 && image::gif_shape(still)->width == 32,
            "a GIF's shape without decoding");
    require(!image::frame_grid(still), "a still GIF has no grid");
    require(!image::frame_grid("\x89PNG\r\n\x1a\n") && !image::frame_grid("GIF89a"), "not a GIF, or cut short");

    const auto moving = make_gif(32, 32, {square_at(0, 0), square_at(12, 12), square_at(24, 24)}, palette);
    auto grid = image::frame_grid(moving);
    require(grid && grid->frames == 3 && grid->total == 3 && grid->duration_ms == 300, "three distinct frames");
    auto decoded = decode_jpeg(grid->jpeg);
    // 2 x 2 cells of 256 px with a 4 px gutter.
    require(decoded.width == 516 && decoded.height == 516, "a near-square grid");
    require(looks_like(decoded.at(10, 10), 0, 0, 0) && looks_like(decoded.at(200, 200), 255, 255, 255),
            "the first frame top left, over white");
    require(looks_like(decoded.at(258, 100), 200, 200, 200), "grey between frames");
    require(looks_like(decoded.at(400, 400), 255, 255, 255), "an empty cell is white");

    std::vector<std::vector<u8>> doubled;
    for (int i = 0; i < 6; ++i) {
        doubled.push_back(square_at(i * 4, 0));
        doubled.push_back(square_at(i * 4, 0));
    }
    grid = image::frame_grid(make_gif(32, 32, doubled, palette));
    require(grid && grid->frames == 6 && grid->total == 12 && grid->indices == std::vector<usize>({0, 2, 4, 6, 8, 10}),
            "repeated frames count once");

    std::vector<std::vector<u8>> many;
    for (int i = 0; i < 30; ++i) many.push_back(square_at(i % 2 == 0 ? 0 : 24, i % 3 * 12));
    grid = image::frame_grid(make_gif(32, 32, many, palette));
    require(grid && grid->frames == 9 && grid->indices.front() == 0 && grid->indices.back() == 29, "nine of thirty, first and last kept");

    // A one-pixel shake stays under the threshold: frames are spread instead.
    std::vector<std::vector<u8>> shake;
    for (int i = 0; i < 8; ++i) {
        std::vector<u8> frame(32 * 32, 0);
        for (int row = 12; row < 16; ++row) {
            for (int column = 12 + i % 2; column < 16 + i % 2; ++column) frame[static_cast<usize>(row * 32 + column)] = 1;
        }
        shake.push_back(std::move(frame));
    }
    grid = image::frame_grid(make_gif(32, 32, shake, palette));
    require(grid && grid->frames == 8, "small motions are not dropped");

    std::vector<u8> clear(32 * 32, 2);
    grid = image::frame_grid(make_gif(32, 32, {clear, square_at(16, 16)}, palette, 10, 2));
    decoded = decode_jpeg(grid->jpeg);
    require(grid && looks_like(decoded.at(100, 100), 255, 255, 255), "transparent pixels become white");

    auto huge = moving;
    huge[6] = '\xa0';
    huge[7] = '\x0f';
    require(!image::frame_grid(huge), "a GIF too large to decode is refused");

    // Cut short anywhere, or with a byte flipped: no crash.
    for (usize length = 0; length < moving.size(); ++length) std::ignore = image::frame_grid(moving.substr(0, length));
    std::mt19937 rng(3);
    for (int i = 0; i < 400; ++i) {
        auto broken = moving;
        broken[rng() % broken.size()] ^= static_cast<char>(1 << (rng() % 8));
        std::ignore = image::frame_grid(broken);
    }
}

lighter::Task<> test_describe() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    StickerConfig config{.enabled = true};
    sticker::Collector collector(config, "data", *db, [](lighter::Task<>) {});
    collector.now = [] { return i64{1'000}; };
    auto state = std::make_shared<test::FakeProvider::State>();
    state->answer = [](const provider::History &, const std::vector<provider::ToolDefinition> &) {
        return std::optional<std::vector<provider::OutputItem>>{{test::text_answer("描述：一只猫在摇头\n标签：摇头、拒绝\n适合：是")}};
    };
    collector.vision = test::fake_provider(state);
    const auto palette = test_palette();
    const auto gif = make_gif(32, 32, {square_at(0, 0), square_at(12, 12), square_at(24, 24)}, palette);
    auto id = db->add_sticker({.sha1 = "g", .path = "stickers/g.gif", .media_type = "image/gif"});
    require(id.has_value(), "add a GIF sticker");
    const auto tagged = co_await collector.describe(**db->sticker_by_id(*id), gif, "");
    require(tagged && tagged->tags == "摇头、拒绝", "the answer is read");
    const auto &parts = state->requests.back().back().parts;
    const auto *picture = std::get_if<provider::ImagePart>(&parts.front());
    const auto *prompt = std::get_if<provider::TextPart>(&parts.back());
    require(parts.size() == 2 && picture && picture->media_type == "image/jpeg" && prompt && prompt->text.contains("3 帧"),
            "a GIF goes as a grid of its frames, and the prompt says so");
    require(!prompt->text.contains("无语、震惊、开心"), "no example tags to copy");

    config.frames_with_original = true;
    std::ignore = co_await collector.describe(**db->sticker_by_id(*id), gif, "");
    require(state->requests.back().back().parts.size() == 3, "the original file can go along");
    config.frames_with_original = false;

    // Tagging respects the background budget and a person's edits.
    require(db->edit_sticker(*id, "人工标签", "人工描述").has_value(), "edit");
    co_await collector.tag(**db->sticker_by_id(*id), gif, "");
    auto edited = **db->sticker_by_id(*id);
    require(edited.tags == "人工标签" && edited.description == "人工描述" && edited.status == "registered" && edited.modified_by == "USER",
            "machine tagging keeps a person's edit but moves the status");
    const auto calls = state->requests.size();
    collector.background_budget = 1;
    std::ignore = db->add_llm_call({.time = 999, .chat = {.type = "learn", .id = 1}, .model = "m", .input_tokens = 5});
    co_await collector.tag(**db->sticker_by_id(*id), gif, "");
    require(state->requests.size() == calls, "no tagging over the background budget");
}

void test_parse_tagging() {
    auto tagged = sticker::parse_tagging("描述：一只猫瞪大眼睛，配字\"啊？\"\n标签：震惊、无语\n适合：是");
    require(tagged && tagged->description == "一只猫瞪大眼睛，配字\"啊？\"" && tagged->tags == "震惊、无语" && tagged->usable,
            "three lines, full-width colons");
    tagged = sticker::parse_tagging("  描述: 聊天记录截图\n标签: 截图\n适合: 否  ");
    require(tagged && !tagged->usable && tagged->description == "聊天记录截图", "ASCII colons and a refusal");
    require(!sticker::parse_tagging("这是一张猫的表情包"), "free text is no answer");
    require(!sticker::parse_tagging("描述：\n标签：无\n适合：是"), "an empty description is no answer");
}

void test_urls() {
    onebot::Segment sticker{.type = "image"};
    sticker.data["url"] = "https://multimedia.nt.qq.com.cn/a";
    sticker.data["sub_type"] = i64{1};
    onebot::Segment photo{.type = "image"};
    photo.data["url"] = "https://multimedia.nt.qq.com.cn/b";
    photo.data["sub_type"] = i64{0};
    onebot::Segment mface{.type = "mface"};
    mface.data["url"] = "https://gxh.vip.qq.com/c";
    onebot::Segment local{.type = "mface"};
    local.data["url"] = "file:///tmp/x";
    require(sticker::sticker_urls({sticker, photo, mface, local, onebot::text_segment("hi")}) ==
                std::vector<std::string>({"https://multimedia.nt.qq.com.cn/a", "https://gxh.vip.qq.com/c"}),
            "stickers and mface with http(s) urls; photos and other urls are skipped");
}

void test_sample() {
    std::vector<store::Sticker> pool;
    for (i64 i = 0; i < 10; ++i) pool.push_back({.id = i, .description = "d" + std::to_string(i), .tags = "t", .seen = 1});
    pool[3].seen = 40;
    std::mt19937 rng(5);
    require(sticker::sample(pool, 20, rng).size() == 10, "never more than the pool");
    int popular = 0;
    for (int i = 0; i < 300; ++i) popular += sticker::sample(pool, 1, rng).front().id == 3;
    // Weight 41 against 9 others of weight 2: about 70% of single picks.
    require(popular > 150, "seen often means picked often");
}

void test_finding() {
    std::vector<store::Sticker> pool{
        {.id = 1, .sha1 = "a", .description = "猫在转圈", .tags = "开心、好耶", .seen = 9},
        {.id = 2, .sha1 = "b", .description = "猫瞪眼，配字\"啊？\"", .tags = "无语、震惊", .seen = 1},
        {.id = 3, .sha1 = "c", .description = "熊猫头摊手", .tags = "无语、无奈", .seen = 5},
    };
    std::mt19937 rng(7);
    const auto found = sticker::match(pool, "无语/震惊", 2, rng);
    require(found.size() == 2 && found[0].id == 2 && found[1].id == 3, "more words matched first, then the more popular");
    require(sticker::match(pool, "晚安", 2, rng).size() == 2, "nothing matched: a sample fills up");

    require(sticker::embedding_text(pool[1]) == "无语、震惊：猫瞪眼，配字\"啊？\"", "embedded as tags and description");

    require(sticker::parse_handle("s12") == 12 && sticker::parse_handle("#s3") == 3 && sticker::parse_handle(" 7 ") == 7 &&
                !sticker::parse_handle("s1x") && !sticker::parse_handle("无语"),
            "sticker numbers");
    pool[1].last_used = 1000;
    require(sticker::render_candidates({pool[1], pool[0]}, 2000) ==
                "s2 标签：无语、震惊｜画面：猫瞪眼，配字\"啊？\"（她一天内发过）\ns1 标签：开心、好耶｜画面：猫在转圈\n",
            "candidates by number, recent ones marked");
}

void test_resend() {
    onebot::Segment sticker{.type = "image"};
    sticker.data["url"] = "https://multimedia.nt.qq.com.cn/a";
    sticker.data["sub_type"] = i64{1};
    const auto again = sticker::resend_segment({sticker});
    require(again && again->type == "image" && onebot::data_string(*again, "file") == "https://multimedia.nt.qq.com.cn/a" &&
                onebot::data_integer(*again, "sub_type") == 1,
            "an image sticker goes again by its url");
    onebot::Segment mface{.type = "mface"};
    mface.data["emoji_id"] = "abc";
    require(sticker::resend_segment({mface}) && sticker::resend_segment({mface})->type == "mface", "an mface goes as it came");
    onebot::Segment photo{.type = "image"};
    photo.data["url"] = "https://multimedia.nt.qq.com.cn/b";
    require(!sticker::resend_segment({photo}) && !sticker::resend_segment({sticker, onebot::text_segment("草")}) &&
                !sticker::resend_segment({onebot::text_segment("6")}),
            "photos, stickers with words and text are not stickers to resend");
}

void test_store_and_pool() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    StickerConfig config{.enabled = true, .max_registered = 2};
    sticker::Collector collector(config, "data", *db, [](lighter::Task<>) {});
    std::vector<i64> ids;
    for (int i = 0; i < 3; ++i) {
        auto id = db->add_sticker({.sha1 = "h" + std::to_string(i), .path = "stickers/x", .media_type = "image/png", .seen = 1 + i});
        require(id.has_value(), "add sticker");
        ids.push_back(*id);
    }
    require((*db->sticker_by_hash("h1"))->id == ids[1] && !*db->sticker_by_hash("nope"), "look up by hash");
    for (int i = 0; i < 2; ++i) collector.file_into_pool(**db->sticker_by_hash("h" + std::to_string(i)));
    require(db->stickers("registered")->size() == 2, "the pool fills up");
    collector.file_into_pool(**db->sticker_by_hash("h2"));
    auto registered = db->stickers("registered");
    require(registered->size() == 2 && registered->front().sha1 == "h2", "a sticker seen more replaces the weakest unsent one");
    require(db->stickers("spare")->front().sha1 == "h0", "the weakest becomes a spare");
    require(db->sticker_seen(ids[0], 99).has_value() && (*db->sticker_by_hash("h0"))->seen == 2, "sightings count");
    require(db->sticker_used(ids[2], 100).has_value() && (*db->sticker_by_hash("h2"))->used == 1, "uses count");
    require(db->describe_sticker(ids[0], "猫", "无语", "rejected").has_value() && (*db->sticker_by_hash("h0"))->status == "rejected",
            "tagging sets the status");
}

} // namespace

int main() {
    try {
        test_parse_tagging();
        test_urls();
        test_sample();
        test_finding();
        test_resend();
        test_store_and_pool();
        test_frames();
        lighter::EventLoop loop;
        auto task = test_describe();
        loop.schedule(task);
        loop.run();
        std::ignore = task.value();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
