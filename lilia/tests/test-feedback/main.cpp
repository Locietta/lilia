#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <lighter/async/io/loop.h>
#include <lighter/async/runtime/task_group.h>

#include <lilia/chat/addressee.h>
#include <lilia/feedback/feedback.h>
#include <lilia/feedback/logistic.h>
#include <lilia/onebot/message.h>
#include <lilia/replay.h>
#include <lilia/store/store.h>
#include <lilia/tests/support/fake_provider.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;

constexpr i64 k_self = 10001;
constexpr i64 k_group = 555;

store::StoredMessage line(i64 row, i64 time, i64 sender, std::string name, std::string text, bool self = false) {
    return {.chat = {.type = "group", .id = k_group},
            .message_id = row * 10,
            .sender_id = self ? k_self : sender,
            .sender_name = std::move(name),
            .time = time,
            .segments_json = R"([{"type":"text","data":{"text":")" + text + R"("}}])",
            .text = text,
            .from_self = self,
            .row_id = row};
}

void test_features() {
    using chat::LineEdges;
    std::vector<store::StoredMessage> rows{
        line(1, 100, 2001, "阿明", "有人玩原神吗"), line(2, 120, 0, "闭锁", "我玩", true), line(3, 130, 2001, "阿明", "你几级了？"),
        line(4, 140, 2002, "小红", "阿明你也玩？"), line(5, 150, 2001, "阿明", "对啊"),    line(6, 160, 2002, "小红", "那你等我"),
        line(7, 170, 2001, "阿明", "草"),
    };
    std::vector<LineEdges> edges(rows.size());
    edges[3].to = {"阿明"};
    edges[3].at_other = true;
    const auto after = chat::line_features(rows, edges, 2);
    require(after[0] == 1 && after[7] == 1 && after[9] == 1 && after[11] == 1 && after[12] == 1 && after[13] == 0.5,
            "right after her, to the one she answered: 你, a question, her share");
    require(std::abs(after[14] - std::log1p(10.0) / std::log1p(86'400.0)) < 1e-9, "seconds since her line, log-scaled");
    const auto named = chat::line_features(rows, edges, 3);
    require(named[4] == 1 && named[6] == 1 && named[11] == 0, "an @ of someone else, and their name");
    const auto dyad = chat::line_features(rows, edges, 6);
    require(dyad[15] == 1 && dyad[10] == 1 && dyad[16] == 1.0 / 6, "a run between two people without her; a reaction");
    const auto first = chat::line_features(rows, edges, 0);
    require(first[14] == 1 && first[8] == 1 && first[13] == 0, "before she ever spoke; asked of everyone");
}

void test_logistic() {
    // Labels drawn from known weights come back near them.
    feedback::Weights truth{};
    truth[0] = -1;
    truth[1] = 3;
    truth[7] = 1;
    std::mt19937 rng(11);
    std::bernoulli_distribution coin(0.5);
    std::uniform_real_distribution<double> unit(0, 1);
    std::vector<feedback::Sample> samples;
    for (int i = 0; i < 4000; ++i) {
        feedback::Sample sample;
        sample.x[0] = 1;
        sample.x[1] = coin(rng);
        sample.x[7] = coin(rng);
        sample.label = unit(rng) < feedback::predict(truth, sample.x) ? 1 : 0;
        samples.push_back(sample);
    }
    const feedback::Weights zero{};
    const auto fitted = feedback::fit_logistic(samples, zero, 0.01);
    require(std::abs(fitted[0] + 1) < 0.2 && std::abs(fitted[1] - 3) < 0.3 && std::abs(fitted[7] - 1) < 0.2, "known weights are recovered");
    require(feedback::log_loss(fitted, samples) < feedback::log_loss(zero, samples), "the fit lowers the loss");
    require(feedback::fit_logistic({}, feedback::bootstrap_weights(), 1) == feedback::bootstrap_weights(), "no data: the prior");
    const std::vector<feedback::Sample> few(samples.begin(), samples.begin() + 5);
    const auto shrunk = feedback::fit_logistic(few, feedback::bootstrap_weights(), 100);
    require(std::abs(shrunk[1] - feedback::bootstrap_weights()[1]) < 0.1, "little data stays near the prior");
    const auto json = feedback::write_weights(fitted);
    require(feedback::read_weights(json) == fitted && !feedback::read_weights("[1, 2]"), "weights round-trip as JSON");
}

void test_parse() {
    const auto hindsight = feedback::parse_hindsight({test::tool_call("record_hindsight", R"({
        "lines": [{"row": "#12", "to": "her"}, {"row": 13, "to": "OTHER"}, {"row": "x", "to": "her"}],
        "replies": [{"row": "#14", "for_her": "no", "answered_by_other": "yes", "stance": "rejection", "contribution": "intrusion"}]})")});
    require(hindsight && hindsight->lines.size() == 2 && hindsight->lines[0].row == 12 && hindsight->lines[0].to == "her" &&
                hindsight->lines[1].to == "unclear",
            "rows as #n or numbers; unknown values are unclear");
    require(hindsight->replies.size() == 1 && hindsight->replies[0].stance == "rejection" && hindsight->replies[0].for_her == "no",
            "her replies");
    require(!feedback::parse_hindsight({test::text_answer("嗯")}), "no tool call");
    require(feedback::reward_of("appreciation", "advance") == 1.5 && feedback::reward_of("rejection", "intrusion") == -2 &&
                feedback::reward_of("?", "?") == 0,
            "rewards");
    require(feedback::level(0.1) == "低" && feedback::level(0.4) == "中" && feedback::level(0.8) == "高", "levels");
}

lighter::Task<> test_judging() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    const store::ChatKey chat{.type = "group", .id = k_group};
    FeedbackConfig config{.enabled = true,
                          .judge_after_seconds = 600,
                          .judge_window_lines = 40,
                          .min_rows_global = 3,
                          .min_rows_group = 3,
                          .min_positives = 1,
                          .refit_every = 1};
    ChatConfig chat_config{.groups = {k_group}};
    auto state = std::make_shared<test::FakeProvider::State>();
    std::string request;
    state->answer = [&request](const provider::History &history, const auto &) -> std::optional<std::vector<provider::OutputItem>> {
        request = std::get<provider::TextPart>(history.back().parts.front()).text;
        return std::vector<provider::OutputItem>{test::tool_call("record_hindsight", R"({
            "lines": [{"row": "#2", "to": "other"}, {"row": "#3", "to": "her"}, {"row": "#5", "to": "other"}],
            "replies": [{"row": "#4", "for_her": "yes", "answered_by_other": "unclear", "stance": "playful", "contribution": "advance"}]})")};
    };
    std::vector<lighter::Task<>> spawned;
    feedback::Feedback judge(config, chat_config, *db, test::fake_provider(state),
                             [&spawned](lighter::Task<> task) { spawned.push_back(std::move(task)); });
    i64 clock = 1'000;
    judge.now = [&clock] { return clock; };

    std::vector<store::StoredMessage> rows{line(1, 900, 2001, "阿明", "在吗"), line(2, 950, 2002, "小红", "阿明你吃了吗"),
                                           line(3, 960, 2001, "阿明", "莉莉娅你吃了吗"), line(4, 970, 0, "闭锁", "吃了", true),
                                           line(5, 980, 2002, "小红", "我也吃了")};
    require(db->add_message(rows[0]).has_value(), "store");
    judge.observe(chat, k_self);
    require(spawned.empty() && db->feedback_progress(chat).value() == 1, "live judging starts from the newest line");
    for (usize i = 1; i < rows.size(); ++i) require(db->add_message(rows[i]).has_value(), "store");
    // Planner log: her message answered #3.
    std::ignore = db->add_planner_turn(
        {.time = 965, .chat = chat, .tool = "reply", .target_message_id = 30, .to_her = "called", .sent_message_ids = "40"});
    judge.observe(chat, k_self);
    require(spawned.empty(), "not yet: her message is too recent");
    clock = 970 + 600;
    judge.observe(chat, k_self);
    require(spawned.size() == 1, "after a while it is judged");
    co_await std::move(spawned.back());
    require(request.contains("#4（回应 #3）") && request.contains("#2、#3、#5"), "the judge sees whom she answered and what to label");
    auto labels = db->addressee_labels(chat::k_feature_version);
    require(labels.has_value() && labels->size() == 3 && labels->at(1).label && !labels->at(0).label, "labels are stored");
    auto effects = db->reply_effects(chat, 0);
    require(effects.has_value() && effects->size() == 1 && effects->front().reward == 1.2 && effects->front().to_her == "called" &&
                effects->front().target_message_id == 30,
            "her message's effect is stored");
    require(db->feedback_progress(chat).value() == 5, "judged through the window");
    require(judge.fitted(chat), "enough labels: a model is fitted");

    // Hints: only for new lines that point at no one, shown only when on.
    std::vector<chat::LineEdges> edges(rows.size());
    auto hint = judge.hints(chat, rows, edges, 2);
    require(hint.shown.empty() && hint.logged.contains("#2 ") && !hint.logged.contains("#4"), "shadow mode: logged, not shown");
    config.hint = true;
    hint = judge.hints(chat, rows, edges, 2);
    require(hint.shown.starts_with("根据这个群以往的情况估计") && hint.shown.contains("#3 "), "shown when on");
    edges[2].to_her = true;
    require(!judge.hints(chat, rows, edges, 2).shown.contains("#3 "), "a line that says whom it is for needs no estimate");
}

void test_gates() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    const store::ChatKey even{.type = "group", .id = 1};
    const store::ChatKey shy{.type = "group", .id = 2};
    const store::ChatKey small{.type = "group", .id = 3};
    // even: 4 rows, 2 to her; shy: 4 rows, 1 to her; small: 1 row.
    const std::vector<std::pair<store::ChatKey, bool>> rows{{even, true}, {even, false}, {even, true}, {even, false}, {shy, true},
                                                            {shy, false}, {shy, false},  {shy, false}, {small, true}};
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> unit(0, 1);
    for (usize i = 0; i < rows.size(); ++i) {
        chat::Features x{};
        for (auto &value : x) value = unit(rng);
        x[0] = 1;
        require(db->add_addressee_label({.chat = rows[i].first,
                                         .message_id = static_cast<i64>(i + 1),
                                         .row_id = static_cast<i64>(i + 1),
                                         .time = static_cast<i64>(100 + i),
                                         .feature_version = chat::k_feature_version,
                                         .features = feedback::write_weights(x),
                                         .label = rows[i].second,
                                         .source = "judge"})
                    .has_value(),
                "label");
    }
    FeedbackConfig config{.enabled = true, .min_rows_global = 6, .min_rows_group = 4, .min_positives = 2};
    const auto samples = feedback::samples_of(db->addressee_labels(chat::k_feature_version).value());
    require(samples.size() == rows.size(), "every label is a sample");
    const auto fitted = feedback::fit_models(samples, config);
    require(fitted.global.fitted && fitted.global.rows == 9 && fitted.global.positives == 4, "enough rows: a global model");
    require(fitted.groups.at(even).fitted && !fitted.groups.at(shy).fitted && !fitted.groups.at(small).fitted,
            "a group needs its rows and its lines to her");
    require(fitted.model_for(even) == fitted.groups.at(even).weights && fitted.model_for(shy) == fitted.global.weights &&
                fitted.model_for({.type = "group", .id = 9}) == fitted.global.weights,
            "a group without its own model uses the global one");
    auto strict = config;
    strict.min_rows_global = 100;
    strict.min_rows_group = 100;
    const auto none = feedback::fit_models(samples, strict);
    require(!none.global.fitted && none.model_for(even) == feedback::bootstrap_weights(), "short of every gate: bootstrap");

    // A refit stores what fit_models fitted, and nothing else.
    auto state = std::make_shared<test::FakeProvider::State>();
    const ChatConfig chat_config{};
    feedback::Feedback refitter(config, chat_config, *db, test::fake_provider(state), [](lighter::Task<>) {});
    refitter.refit();
    const auto stored = db->addressee_models().value();
    require(stored.size() == 2, "the global model and one group's");
    for (const auto &model : stored) {
        const auto &expected = model.chat.type.empty() ? fitted.global : fitted.groups.at(model.chat);
        require(model.chat.type.empty() || model.chat == even, "only the group past its gates");
        require(feedback::read_weights(model.weights) == expected.weights && model.rows == static_cast<i64>(expected.rows) &&
                    model.positives == static_cast<i64>(expected.positives),
                "the stored model is the one fit_models fitted");
    }
    for (const auto &chat : {even, shy, small}) {
        require(refitter.model_for(chat) == fitted.model_for(chat), "the refit uses what --eval scores");
    }
}

void test_sticker_rewards() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    const store::ChatKey chat{.type = "group", .id = k_group};
    const auto effect = [&](i64 message, std::string kind, i64 sticker, double reward) {
        require(db->add_reply_effect({.chat = chat,
                                      .her_message_id = message,
                                      .kind = std::move(kind),
                                      .sticker_id = sticker,
                                      .for_her = "yes",
                                      .answered_by_other = "no",
                                      .stance = "neutral",
                                      .contribution = "maintain",
                                      .reward = reward,
                                      .time = message})
                    .has_value(),
                "effect");
    };
    effect(1, "sticker", 7, 1.2);
    effect(2, "sticker", 7, -0.5);
    effect(3, "text", 0, 1.0);
    const auto rewards = db->sticker_rewards();
    require(rewards.has_value() && rewards->size() == 1, "only stickers count");
    require(rewards->at(7).judged == 2 && rewards->at(7).welcomed == 1, "two sends of one sticker");
    // Judged again: the effect is replaced, not counted twice.
    effect(2, "sticker", 7, 0.3);
    const auto again = db->sticker_rewards();
    require(again.has_value() && again->at(7).judged == 2 && again->at(7).welcomed == 2, "a replaced effect is counted once");
}

void test_probe_picks() {
    const auto label = [](i64 row, bool to_her, std::string type = "group") {
        return store::AddresseeLabel{
            .chat = {.type = std::move(type), .id = k_group}, .message_id = row * 10, .row_id = row, .label = to_her};
    };
    const std::vector<store::AddresseeLabel> labels{
        label(1, false), label(2, true), label(3, false), label(4, false), label(5, false, "private"), label(6, true)};
    const auto rows = [](const std::vector<store::AddresseeLabel> &picked) {
        std::vector<i64> out;
        for (const auto &one : picked) out.push_back(one.row_id);
        return out;
    };
    require(rows(pick_probe_labels(labels, 0)) == std::vector<i64>{1, 2, 3, 4, 6}, "no limit: every group line");
    require(rows(pick_probe_labels(labels, 3)) == std::vector<i64>{2, 4, 6}, "every line to her, then the newest others");
    require(rows(pick_probe_labels(labels, 1)) == std::vector<i64>{2, 6}, "lines to her are never cut");
}

void test_probe_flow() {
    auto source = store::Store::open_memory();
    require(source.has_value(), "open");
    const auto with = [](store::StoredMessage row, const onebot::Message &segments) {
        row.segments_json = glz::write_json(segments).value_or("[]");
        return row;
    };
    auto notice = line(5, 150, 0, "群通知", "小明戳了戳莉莉娅");
    notice.segments_json = "[]";
    const std::vector<store::StoredMessage> rows{
        with(line(1, 100, 2001, "阿明", "在吗"), {onebot::at_segment(k_self), onebot::text_segment(" 在吗")}),
        line(2, 110, 0, "莉莉娅", "在", true),
        line(3, 120, 2002, "小红", "早"),
        with(line(4, 130, 2001, "阿明", "你在干嘛"), {onebot::reply_segment(20), onebot::text_segment("你在干嘛")}),
        notice,
        line(6, 160, 2002, "小红", "莉莉娅也来"),
        line(7, 170, 0, "莉莉娅", "来了", true),
        line(8, 180, 2001, "阿明", "好"),
    };
    for (const auto &row : rows) require(source->add_message(row).has_value(), "stored");
    ChatConfig config;
    config.groups = {k_group};
    config.aliases = {"莉莉娅"};
    const auto flow = probe_flow(rows, k_self, config, *source);
    std::vector<i64> unseen;
    for (const auto &row : flow.unseen) unseen.push_back(row.row_id);
    require(unseen == std::vector<i64>{4, 6, 8}, "the last three of others' lines are new; notices and hers are not");
    require(flow.addressed == 40, "a quote of her line addresses her; the older @ is not new");
    require(flow.named == 60, "her name");
    require(self_id_of(*source, {k_group}) == k_self, "her id from her lines");
}

} // namespace

int main() {
    try {
        test_features();
        test_logistic();
        test_parse();
        test_gates();
        test_sticker_rewards();
        test_probe_picks();
        test_probe_flow();
        lighter::EventLoop loop;
        auto task = test_judging();
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
