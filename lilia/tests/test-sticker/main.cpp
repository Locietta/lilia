#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <lilia/onebot/message.h>
#include <lilia/sticker/sticker.h>
#include <lilia/store/store.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;

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

void test_sample_and_render() {
    std::vector<store::Sticker> pool;
    for (i64 i = 0; i < 10; ++i) pool.push_back({.id = i, .description = "d" + std::to_string(i), .tags = "t", .seen = 1});
    pool[3].seen = 40;
    std::mt19937 rng(5);
    require(sticker::sample(pool, 20, rng).size() == 10, "never more than the pool");
    int popular = 0;
    for (int i = 0; i < 300; ++i) popular += sticker::sample(pool, 1, rng).front().id == 3;
    // Weight 41 against 9 others of weight 2: about 70% of single picks.
    require(popular > 150, "seen often means picked often");
    const auto text = sticker::render_candidates({pool[0], pool[3]});
    require(text == "1. 标签：t｜画面：d0\n2. 标签：t｜画面：d3\n", "numbered candidates");
    const auto request = sticker::build_pick_request("P", {"a: 抽卡又歪了"}, "无语：吐槽运气", {pool[0]});
    std::string user;
    for (const auto &part : request.history.back().parts) {
        if (const auto *text_part = std::get_if<provider::TextPart>(&part)) user += text_part->text;
    }
    require(user.find("a: 抽卡又歪了") != std::string::npos && user.find("她想表达的意思：无语：吐槽运气") != std::string::npos &&
                user.find("1. 标签：t｜画面：d0") != std::string::npos && request.tools.front().name == "pick_sticker",
            "the pick request");
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
        test_sample_and_render();
        test_store_and_pool();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
