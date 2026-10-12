#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <lighter/async/io/loop.h>
#include <lighter/async/runtime/task_group.h>

#include <lilia/chat/addressee.h>
#include <lilia/feedback/feedback.h>
#include <lilia/feedback/logistic.h>
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

} // namespace

int main() {
    try {
        test_features();
        test_logistic();
        test_parse();
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
