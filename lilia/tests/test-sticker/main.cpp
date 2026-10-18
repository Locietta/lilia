#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <lighter/async/io/loop.h>

#include <lilia/onebot/message.h>
#include <lilia/sticker/sticker.h>
#include <lilia/store/store.h>
#include <lilia/tests/support/fake_provider.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;

void test_usage_helpers() {
    using sticker::ContextLine;
    const std::vector<ContextLine> lines{
        {.sender = 1, .name = "小明同学", .text = "小红你看 https://b23.tv/abc 这个"},
        {.sender = 2, .name = "小红", .text = "@小明同学 笑死 我的号是 123456789"},
        {.sender = 1, .name = "小明同学", .text = ""},
        {.text = "草", .self = true},
        {.sender = 3, .name = "阿", .text = std::string(100, 'x')},
    };
    const auto snippet = sticker::usage_snippet(lines, 2, "莉莉娅");
    require(snippet == "群友1: 群友2你看 [链接] 这个\n群友2: @某人 笑死 我的号是 [数字]\n群友1: [这张表情包]\n[SELF] 莉莉娅: 草\n群友3: " +
                           std::string(80, 'x') + "…\n",
            "speakers anonymised, names, links, mentions and numbers scrubbed, the sticker marked, long lines cut");

    require(!sticker::usage_due(2, 0) && sticker::usage_due(3, 0) && !sticker::usage_due(5, 3) && sticker::usage_due(6, 3) &&
                sticker::usage_due(30, 6) && !sticker::usage_due(200, 96),
            "inference at 3, 6, 12, 24, 48 and 96 snippets");

    auto usage = sticker::parse_usage(
        {test::tool_call("record_sticker_usage",
                         R"({"usage": "当被说中时发，表示心虚", "tags": "心虚，被戳穿,尴尬、装傻、无语、多余", "no_info": false})")});
    require(usage && usage->usage == "当被说中时发，表示心虚" && usage->tags == "心虚、被戳穿、尴尬、装傻、无语" && !usage->no_info,
            "the usage, at most five tags, one separator");
    usage = sticker::parse_usage({test::tool_call("record_sticker_usage", R"({"usage": "", "tags": "", "no_info": false})")});
    require(usage && usage->no_info, "an empty usage is no information");
    require(!sticker::parse_usage({test::text_answer("当……时发")}), "no tool call, no answer");

    store::Sticker known{.id = 5, .description = "猫摇头", .tags = "摇头", .usage = "当拒绝时发"};
    const std::vector<store::StickerEvidence> evidence{{.snippet = "群友1: 去吗\n群友2: [这张表情包]\n"},
                                                       {.snippet = "[SELF] 莉莉娅: [这张表情包]\n群友1: 啥意思\n", .own = true}};
    const auto request = sticker::build_usage_request("画面{description}标签{tags}{previous}群{groups}{contexts}{previous_instruction}",
                                                      known, evidence, 2, "莉莉娅");
    const auto *text = std::get_if<provider::TextPart>(&request.history.back().parts.front());
    require(text && text->text.contains("画面猫摇头") && text->text.contains("上一次总结的用法（仅供参考）：当拒绝时发") &&
                text->text.contains("群2") && text->text.contains("【片段 2（她自己发的）】") && text->text.contains("以多数片段为准") &&
                request.tools.size() == 1 && request.tools[0].name == "record_sticker_usage",
            "the request carries the picture, the previous usage and the snippets");

    require(sticker::render_candidates({known}, 0) == "s5 标签：摇头｜画面：猫摇头｜常用于：当拒绝时发\n" &&
                sticker::render_candidates({known}, 0, false) == "s5 标签：摇头｜画面：猫摇头\n",
            "candidates show the usage, unless it is switched off");
    require(sticker::embedding_inputs(known, "data", false, true) == std::vector<std::string>({"摇头：猫摇头", "用法：当拒绝时发"}) &&
                sticker::embedding_inputs(known, "data", false, false).size() == 1 &&
                sticker::lexical_text(known, true) == "摇头 猫摇头 当拒绝时发",
            "searches can use the usage");
}

void test_usage_store() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    auto id = db->add_sticker({.sha1 = "u", .path = "stickers/u.png", .media_type = "image/png", .tags = "猫", .status = "registered"});
    require(id.has_value(), "add");
    for (int i = 0; i < 5; ++i) {
        auto after = db->add_sticker_evidence(*id, {.type = "group", .id = 7 + i % 2}, "片段" + std::to_string(i), i == 4, 100 + i, 3);
        require(after.has_value() && after->usage_count == i + 1, "snippets are counted");
    }
    auto kept = db->sticker_evidence(*id, 10);
    require(kept.has_value() && kept->size() == 3 && kept->front().snippet == "片段2" && kept->back().own && kept->back().chat.id == 7,
            "only the newest snippets stay, oldest first");
    require(db->set_sticker_usage(*id, "当……时发", "摇头、拒绝", 5).has_value(), "set usage");
    auto sticker = **db->sticker_by_id(*id);
    require(sticker.usage == "当……时发" && sticker.tags == "摇头、拒绝" && sticker.tags_by == "USAGE" && sticker.usage_inferred_count == 5,
            "usage and its tags are stored");
    require(db->set_sticker_usage(*id, std::nullopt, std::nullopt, 6).has_value(), "no information");
    sticker = **db->sticker_by_id(*id);
    require(sticker.usage == "当……时发" && sticker.usage_inferred_count == 6, "no information keeps the usage, counts the inference");
    require(db->edit_sticker(*id, "人工", "人工描述").has_value() && db->set_sticker_usage(*id, "新用法", "机器标签", 7).has_value(),
            "edit");
    sticker = **db->sticker_by_id(*id);
    require(sticker.tags == "人工" && sticker.usage == "新用法", "a person's tags stay; the usage is still learned");
    auto forgotten = db->delete_sticker_evidence_of_group(8);
    require(forgotten.has_value() && *forgotten == 1 && db->sticker_evidence(*id, 10)->size() == 2, "forget a group");
    require(db->describe_sticker(*id, "x", "y", "rejected").has_value() && db->sticker_evidence(*id, 10)->empty(),
            "a rejected sticker loses its snippets");
}

onebot::MessageEvent group_message(i64 group, i64 user, std::string name, onebot::Message message) {
    return {.message_type = "group",
            .user_id = user,
            .group_id = group,
            .sender = {.user_id = user, .nickname = std::move(name)},
            .message = std::move(message)};
}

lighter::Task<> test_sightings() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    StickerConfig config{.enabled = true, .exclude_groups = {99}};
    std::vector<lighter::Task<>> spawned;
    sticker::Collector collector(config, "data", *db, [&spawned](lighter::Task<> task) { spawned.push_back(std::move(task)); });
    i64 clock = 1'000;
    collector.now = [&clock] { return clock; };
    collector.self_name = "莉莉娅";
    auto state = std::make_shared<test::FakeProvider::State>();
    state->answer = [](const provider::History &, const std::vector<provider::ToolDefinition> &) {
        return std::optional<std::vector<provider::OutputItem>>{
            {test::tool_call("record_sticker_usage", R"({"usage": "当拒绝别人时发", "tags": "拒绝、不要", "no_info": false})")}};
    };
    collector.helper = test::fake_provider(state);
    auto id = db->add_sticker({.sha1 = "s", .path = "stickers/s.png", .media_type = "image/png", .tags = "猫", .status = "registered"});
    require(id.has_value(), "add");

    onebot::Segment sticker{.type = "image"};
    sticker.data["url"] = "https://multimedia.nt.qq.com.cn/s";
    sticker.data["sub_type"] = i64{1};
    for (int round = 0; round < 3; ++round) {
        clock += 700; // older lines have left the group's recent ones
        collector.observe(group_message(7, 2001, "阿明", {onebot::text_segment("一起去吃饭吗")}), 10);
        ++clock;
        collector.observe(group_message(7, 2002, "小红", {sticker}), 10);
        require(collector.sightings.size() == 1 && collector.sightings.back().before.size() == 1, "a sighting opens with the line before");
        // collect() would download it and fill in the id.
        *collector.sightings.back().sticker_id = *id;
        collector.observe(group_message(7, 2001, "阿明", {onebot::text_segment("好吧")}), 10);
        collector.observe(group_message(99, 2003, "外人", {onebot::text_segment("别的群")}), 10);
        ++clock;
        require(collector.sightings.size() == 1, "it waits for more lines after it (excluded groups do not count)");
        clock += 200;
        collector.observe(group_message(8, 2004, "路人", {onebot::text_segment("另一个群")}), 10);
        require(collector.sightings.empty(), "after its window any group's message closes it");
    }
    auto evidence = db->sticker_evidence(*id, 20);
    require(evidence.has_value() && evidence->size() == 3 &&
                evidence->front().snippet == "群友1: 一起去吃饭吗\n群友2: [这张表情包]\n群友1: 好吧\n",
            "each sighting leaves an anonymised snippet");
    require(spawned.size() == 4, "three downloads, then one inference at three snippets");
    co_await std::move(spawned.back());
    auto learned = **db->sticker_by_id(*id);
    require(learned.usage == "当拒绝别人时发" && learned.tags == "拒绝、不要" && learned.usage_inferred_count == 3 &&
                state->requests.size() == 1,
            "the usage is learned from the snippets");

    // Her own sticker: the reactions after it are its evidence.
    clock += 700;
    collector.observe_own_text(7, "我不去");
    collector.observe_own(7, *id);
    collector.observe(group_message(7, 2001, "阿明", {onebot::text_segment("啥意思")}), 10);
    clock += 200;
    collector.observe(group_message(8, 2004, "路人", {onebot::text_segment("嗯")}), 10);
    evidence = db->sticker_evidence(*id, 20);
    require(evidence->size() == 4 && evidence->back().own &&
                evidence->back().snippet.contains("[SELF] 莉莉娅: 我不去\n[SELF] 莉莉娅: [这张表情包]\n群友1: 啥意思"),
            "her own sticker is a sighting with the reactions to it");
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
    const std::string gif = "GIF89a";
    auto id = db->add_sticker({.sha1 = "g", .path = "stickers/g.gif", .media_type = "image/gif"});
    require(id.has_value(), "add a GIF sticker");
    const auto tagged = co_await collector.describe(**db->sticker_by_id(*id), gif, "");
    require(tagged && tagged->tags == "摇头、拒绝", "the answer is read");
    const auto &parts = state->requests.back().back().parts;
    const auto *picture = std::get_if<provider::ImagePart>(&parts.front());
    const auto *prompt = std::get_if<provider::TextPart>(&parts.back());
    require(parts.size() == 2 && picture && picture->media_type == "image/gif" && prompt, "a GIF goes whole");
    require(!prompt->text.contains("无语、震惊、开心"), "no example tags to copy");

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
    for (i64 i = 0; i < 10; ++i) pool.push_back({.id = i, .description = "d" + std::to_string(i), .tags = "t", .heat = 1});
    pool[3].heat = 40;
    std::mt19937 rng(5);
    require(sticker::sample(pool, 20, rng).size() == 10, "never more than the pool");
    int popular = 0;
    for (int i = 0; i < 300; ++i) popular += sticker::sample(pool, 1, rng).front().id == 3;
    // Weight 41 against 9 others of weight 2: about 70% of single picks.
    require(popular > 150, "hot means picked often");
    // Sends of sticker 3 went down badly every time: mean 1/10, a fifth of its weight.
    const sticker::Weighing weighing{.rewards = {{3, {.judged = 8, .welcomed = 0}}}};
    popular = 0;
    for (int i = 0; i < 300; ++i) popular += sticker::sample(pool, 1, rng, weighing).front().id == 3;
    // Weight 8.2 against 18: about 31%.
    require(popular > 60 && popular < 130, "a sticker that goes down badly comes up less");
}

void test_rewards() {
    const sticker::Rewards rewards{{7, {.judged = 2, .welcomed = 2}}, {8, {.judged = 3, .welcomed = 0}}};
    require(sticker::reward_mean(rewards, 9) == 0.5, "never judged: even odds");
    require(std::abs(sticker::reward_mean(rewards, 7) - 0.75) < 1e-12 && std::abs(sticker::reward_mean(rewards, 8) - 0.2) < 1e-12,
            "the mean of the Beta(1, 1) posterior");
    require(sticker::reward_boost(0.5, 0.02) == 0 && std::abs(sticker::reward_boost(1, 0.02) - 0.06) < 1e-12 &&
                std::abs(sticker::reward_boost(0, 0.02) + 0.06) < 1e-12,
            "a boost within ±0.06 at the default weight");
    require(std::abs(sticker::reward_boost(0.75, 0.02) - 0.03) < 1e-12, "two good sends of two: +0.03");
}

void test_finding() {
    std::vector<store::Sticker> pool{
        {.id = 1, .sha1 = "a", .description = "猫在转圈", .tags = "开心、好耶", .heat = 9},
        {.id = 2, .sha1 = "b", .description = "猫瞪眼，配字\"啊？\"", .tags = "无语、震惊", .heat = 1},
        {.id = 3, .sha1 = "c", .description = "熊猫头摊手", .tags = "无语、无奈", .heat = 5},
        {.id = 4, .sha1 = "d", .description = "狗狗无语", .tags = "无奈", .heat = 2},
    };
    std::mt19937 rng(7);
    const auto found = sticker::match(pool, "无语/震惊", 3, rng);
    require(found.size() == 3 && found[0].id == 2 && found[1].id == 3 && found[2].id == 4, "more words matched first, then the hotter");
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
    StickerConfig config{.enabled = true, .max_registered = 2, .half_life_days = 14};
    sticker::Collector collector(config, "data", *db, [](lighter::Task<>) {});
    constexpr i64 k_day = 86'400;
    i64 clock = 100 * k_day;
    collector.now = [&clock] { return clock; };
    std::map<std::string, i64> ids;
    const auto add = [&](std::string sha1, double heat) {
        auto id = db->add_sticker({.sha1 = sha1, .path = "stickers/x", .media_type = "image/png", .heat = heat, .heat_time = clock});
        require(id.has_value(), "add sticker");
        ids[sha1] = *id;
        return **db->sticker_by_id(*id);
    };
    const auto members = [&] {
        std::set<std::string> out;
        for (const auto &member : *db->stickers("registered")) out.insert(member.sha1);
        return out;
    };
    const auto h0 = add("h0", 1);
    const auto h1 = add("h1", 2);
    require((*db->sticker_by_hash("h1"))->id == ids["h1"] && !*db->sticker_by_hash("nope"), "look up by hash");
    collector.file_into_pool(h0);
    collector.file_into_pool(h1);
    require(members() == std::set<std::string>{"h0", "h1"}, "the pool fills up");
    collector.file_into_pool(add("h2", 3));
    require(members() == std::set<std::string>{"h1", "h2"} && db->stickers("spare")->front().sha1 == "h0",
            "a hotter sticker takes the place of the coldest, which becomes a spare");
    collector.file_into_pool(add("cold", 1.5));
    require(members() == std::set<std::string>{"h1", "h2"}, "a colder one stays out");
    collector.file_into_pool(add("near", 2.5));
    require(members() == std::set<std::string>{"h1", "h2"}, "hotter by less than a sighting stays out");

    // Every member has been sent: a hotter newcomer still gets in.
    require(db->sticker_used(ids["h1"], clock, 14).has_value() && db->sticker_used(ids["h2"], clock, 14).has_value(), "sent");
    require((*db->sticker_by_hash("h1"))->used == 1 && (*db->sticker_by_hash("h1"))->heat == 4, "a send counts and adds 2");
    collector.file_into_pool(add("h3", 6));
    require(members() == std::set<std::string>{"h2", "h3"}, "sent once is no shield");

    // Heat decays: two months on, h2 (5) and h3 (6) are cold next to a
    // sticker seen three times this week.
    clock += 60 * k_day;
    const auto h2 = **db->sticker_by_hash("h2");
    require(std::abs(store::current_heat(h2, clock, 14) - 5 * std::exp2(-60.0 / 14)) < 1e-9 && store::current_heat(h2, clock, 0) == 5,
            "heat halves every half-life; 0: never");
    auto h4 = add("h4", 0);
    for (int i = 0; i < 3; ++i) {
        auto seen = db->sticker_seen(h4.id, clock, 14);
        require(seen.has_value(), "seen");
        h4 = *seen;
    }
    require(h4.seen == 4 && h4.heat == 3 && h4.heat_time == clock, "each sighting adds 1");
    collector.file_into_pool(h4);
    require(members() == std::set<std::string>{"h3", "h4"}, "the member unseen for long loses to the recent one");
    clock += 14 * k_day;
    const auto again = db->sticker_seen(h4.id, clock, 14);
    require(again.has_value() && again->heat == 2.5, "decayed before it grows: 3 halved, plus 1");

    // A sticker a person edited is never displaced, however cold.
    require(db->edit_sticker(ids["h3"], "人工", "人工").has_value(), "edit");
    collector.file_into_pool(add("h5", 9));
    require(members() == std::set<std::string>{"h3", "h5"}, "an edited sticker stays; the other one makes room");
    collector.file_into_pool(add("h6", 8));
    require(members() == std::set<std::string>{"h3", "h5"}, "with only colder newcomers nothing moves");
    require(db->describe_sticker(ids["h0"], "猫", "无语", "rejected").has_value() && (*db->sticker_by_hash("h0"))->status == "rejected",
            "tagging sets the status");
}

} // namespace

int main() {
    try {
        test_parse_tagging();
        test_urls();
        test_sample();
        test_rewards();
        test_finding();
        test_resend();
        test_store_and_pool();
        lighter::EventLoop loop;
        test_usage_helpers();
        test_usage_store();
        auto task = test_describe();
        loop.schedule(task);
        loop.run();
        std::ignore = task.value();
        auto sightings = test_sightings();
        loop.schedule(sightings);
        loop.run();
        std::ignore = sightings.value();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
