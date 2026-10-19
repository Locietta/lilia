#include "session.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <optional>
#include <set>
#include <variant>

#include <lilia/json.h>

namespace lilia::chat {

namespace {

/// "#18345", "18345" or "#18345 小明: ..." -> the row shown under that number.
const store::StoredMessage *find_target(std::string_view text, const std::vector<store::StoredMessage> &rows) {
    text = trim(text);
    if (text.starts_with('#')) text.remove_prefix(1);
    i64 row = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), row);
    if (ec != std::errc{} || end == text.data()) return nullptr;
    const auto it = std::ranges::find(rows, row, &store::StoredMessage::row_id);
    return it == rows.end() ? nullptr : &*it;
}

provider::SchemaProperty property(std::string type, std::string description) {
    return {.type = std::move(type), .description = std::move(description)};
}

std::string render_tail(const PlannerTail &tail, const ChatInfo &info) {
    // Without it, notes right after her own last line read as part of it
    // ("#42 末尾混入了系统的记忆摘要", replays of 2026-10-12).
    std::string text = "——以上是聊天记录；下面是给你的参考，不是群里的消息——\n";
    if (tail.first_new_row > 0) text += std::format("#{} 起是新消息。\n", tail.first_new_row);
    for (const auto &fact : tail.facts) text += fact + "\n";
    if (!tail.hint.empty()) text += tail.hint + "\n";
    if (!tail.presence.empty()) text += tail.presence + "\n";
    if (!tail.self_notes.empty()) {
        text +=
            std::format("\n{}以前说过的关于她自己的事（她的意图要和这些一致，同一件事以日期新的为准）：\n{}", info.name, tail.self_notes);
    }
    if (!tail.jargon.empty()) text += "\n" + tail.jargon;
    if (!tail.recall.empty()) text += "\n聊天回想（以前聊过的相关内容，内部参考；自然相关时才用，别生硬复述）：\n" + tail.recall;
    if (!text.empty() && !text.ends_with('\n')) text += '\n';
    text += clock_line(tail.now, tail.utc_offset_minutes);
    return text;
}

} // namespace

std::string session_line(const store::StoredMessage &message, i32 utc_offset_minutes, const LineEdges *edges) {
    if (!edges || is_notice(message)) return std::format("#{} {}\n", message.row_id, render_line(message, utc_offset_minutes));
    // "[21:05] 小明(20001): " with the addressee after the speaker.
    auto bare = message;
    bare.text.clear();
    auto speaker = render_line(bare, utc_offset_minutes);
    if (speaker.ends_with(": ")) speaker.resize(speaker.size() - 2);
    return std::format("#{} {}{}: {}\n", message.row_id, speaker, edge_label(*edges), text_with_quote(message, *edges));
}

provider::History render_session(std::string_view prompt, const ChatInfo &info, const std::vector<store::StoredMessage> &rows,
                                 const std::deque<PlannerRecord> &records, const PlannerTail &tail, const std::vector<LineEdges> &edges,
                                 std::string_view earlier, const Pictures &pictures) {
    const auto line = [&](usize index) {
        return session_line(rows[index], tail.utc_offset_minutes, index < edges.size() ? &edges[index] : nullptr);
    };
    provider::History history;
    provider::append_system(history, std::string(prompt));

    provider::Item user{.role = provider::Role::USER};
    std::string lines = chat_place(info) + "\n\n";
    if (!earlier.empty()) lines += std::format("更早的聊天（已经不在下面的记录里，这是当时的摘要）：\n{}\n", earlier);
    lines += "聊天记录（行首是消息编号）：\n";
    const auto flush_lines = [&] {
        if (lines.empty()) return;
        user.parts.push_back(provider::TextPart{.text = std::exchange(lines, {})});
    };
    const auto flush_user = [&] {
        flush_lines();
        if (!user.parts.empty()) history.push_back(std::exchange(user, provider::Item{.role = provider::Role::USER}));
    };

    usize next = 0;
    // A row's pictures come right before its line, so the part that ends the
    // cached prefix is always text.
    const auto add_row = [&] {
        if (const auto found = pictures.find(rows[next].row_id); found != pictures.end()) {
            flush_lines();
            for (const auto &picture : found->second) user.parts.push_back(picture);
        }
        lines += line(next++);
    };
    for (const auto &record : records) {
        while (next < rows.size() && rows[next].row_id <= record.after_row) add_row();
        flush_user();
        for (const auto &item : record.output) provider::append_output_item(history, item);
        // The results first: Anthropic wants them right after the calls.
        for (const auto &outcome : record.outcomes) user.parts.push_back(outcome);
    }
    while (next < rows.size()) add_row();
    flush_lines();
    // Everything so far is the stable part; the tail changes every round.
    if (!user.parts.empty()) {
        if (auto *text = std::get_if<provider::TextPart>(&user.parts.back())) text->cache = true;
    }
    user.parts.push_back(provider::TextPart{.text = render_tail(tail, info)});
    history.push_back(std::move(user));
    return history;
}

void trim_records(std::deque<PlannerRecord> &records, i64 anchor_row, usize keep_cycles) {
    std::erase_if(records, [&](const PlannerRecord &record) { return record.after_row <= anchor_row; });
    std::set<u64> cycles;
    for (const auto &record : records) cycles.insert(record.cycle);
    while (cycles.size() > keep_cycles) cycles.erase(cycles.begin());
    for (auto &record : records) {
        if (cycles.contains(record.cycle)) continue;
        std::erase_if(record.output,
                      [](const provider::OutputItem &item) { return std::holds_alternative<provider::AssistantMessageItem>(item); });
    }
    // A round that was only analysis (she stayed silent) has nothing left.
    std::erase_if(records, [](const PlannerRecord &record) { return record.output.empty(); });
}

std::string render_rounds(const std::deque<PlannerRecord> &records, i64 through_row) {
    constexpr usize k_analysis_bytes = 300;
    constexpr usize k_result_bytes = 120;
    std::string out;
    for (const auto &record : records) {
        if (record.after_row > through_row) continue;
        std::string analysis;
        for (const auto &item : record.output) {
            if (const auto *message = std::get_if<provider::AssistantMessageItem>(&item)) {
                for (const auto &part : message->parts) analysis += part.text;
            }
        }
        std::string results;
        for (const auto &outcome : record.outcomes) {
            results += (results.empty() ? "" : "；") + bounded_utf8(trim(outcome.receipt), k_result_bytes);
        }
        if (trim(analysis).empty() && results.empty()) continue;
        out += std::format("- 看到 #{} 时：{}{}{}\n", record.after_row, bounded_utf8(trim(analysis), k_analysis_bytes),
                           results.empty() ? "" : " → ", results);
    }
    return out;
}

std::deque<PlannerRecord> restore_records(const std::vector<store::PlannerTurn> &turns, const std::vector<store::StoredMessage> &rows,
                                          i64 anchor_row) {
    std::deque<PlannerRecord> records;
    std::string calls;
    std::string analysis;
    const auto close = [&] {
        if (records.empty()) return;
        std::string text = analysis;
        if (!calls.empty()) text += std::format("{}（重启前的记录。当时：{}）", text.empty() ? "" : "\n", calls);
        if (!text.empty()) records.back().output.push_back(provider::AssistantMessageItem{.parts = {{.text = std::move(text)}}});
        if (records.back().output.empty()) records.pop_back();
        calls.clear();
        analysis.clear();
    };
    u64 last_cycle = 0;
    i64 last_round = -1;
    u64 cycle = 0;
    for (const auto &turn : turns) {
        if (turn.shown_row <= anchor_row || rows.empty() || turn.shown_row > rows.back().row_id) continue;
        if (records.empty() || turn.cycle != last_cycle || turn.round != last_round) {
            close();
            if (records.empty() || turn.cycle != last_cycle) ++cycle;
            records.push_back({.after_row = turn.shown_row, .cycle = cycle});
            analysis = turn.analysis;
            last_cycle = turn.cycle;
            last_round = turn.round;
        }
        if (turn.tool.empty()) continue;
        std::string call = turn.tool;
        if (const auto target = std::ranges::find(rows, turn.target_message_id, &store::StoredMessage::message_id);
            turn.target_message_id != 0 && target != rows.end()) {
            call += std::format(" #{}", target->row_id);
        }
        if (!turn.argument.empty()) call += std::format("（{}）", turn.argument);
        if (turn.tool == "stay_silent" && !turn.reason.empty()) call += std::format("（{}）", turn.reason);
        calls += (calls.empty() ? "" : "、") + call;
    }
    close();
    return records;
}

std::vector<provider::ToolDefinition> planner_tools(const PlannerToolSet &set) {
    std::vector<provider::ToolDefinition> tools;
    tools.push_back({
        .name = "reply",
        .description = "让她说话，回应其中一条消息",
        .input_schema = {.properties = {{"target", property("string", "要回应的消息编号，例如 \"#18345\"")},
                                        {"to_her", property("string", "那条消息是不是在对她说：called | likely | group | unsure | no")},
                                        {"quote", property("boolean", "是否引用那条消息")},
                                        {"style", property("string", "short | normal | long")},
                                        {"intent", property("string", "一句话描述她想表达的意思，不是台词")}},
                         .required = {"target", "to_her", "style", "intent"}},
    });
    if (set.stickers) {
        // Tools of their own, like MaiBot's send_emoji: buried in reply, a
        // sticker was never picked. The planner sees the candidates and
        // chooses itself.
        tools.push_back({
            .name = "find_sticker",
            .description = "找几张她能发的表情包看看（编号、标签和画面），然后可以用 send_sticker 发其中一张",
            .input_schema = {.properties = {{"feeling",
                                             property("string", "想表达的感觉或反应，比如\"笑死\"\"无语\"\"震惊\"\"好耶\"\"晚安\"")}},
                             .required = {"feeling"}},
        });
        tools.push_back({
            .name = "send_sticker",
            .description = "发一张表情包，用来表达情绪或反应",
            .input_schema = {.properties = {{"sticker", property("string", "find_sticker 找到的表情包编号，例如 \"s12\"")}},
                             .required = {"sticker"}},
        });
    }
    tools.push_back({
        .name = "repeat",
        .description = "复读：把一条群友的短消息或表情包原样再发一遍（QQ 群里的玩梗习惯）",
        .input_schema = {.properties = {{"target", property("string", "要复读的消息编号，例如 \"#18345\"")}}, .required = {"target"}},
    });
    tools.push_back({
        .name = "wait",
        .description = "先等一会儿再看：等完之后新消息会出现在聊天记录里，再决定下一步",
        .input_schema = {.properties = {{"seconds", property("integer", "等多少秒，一般 5 到 20")},
                                        {"reason", property("string", "为什么等，简短")}},
                         .required = {"seconds"}},
    });
    if (set.poke) {
        tools.push_back({
            .name = "poke",
            .description = "戳一戳群友（QQ 的戳一戳）：被人戳了想戳回去，或者想逗一下谁",
            .input_schema = {.properties = {{"target", property("string", "那人发的某条消息的编号，或者「戳了戳她」那条群通知的编号")}},
                             .required = {"target"}},
        });
    }
    if (set.memory) {
        tools.push_back({
            .name = "search_history",
            .description = "搜索这个群以前的聊天记录，结果会告诉你，然后你再决定下一步",
            .input_schema = {.properties = {{"query",
                                             property("string", "要找的内容，写成一句完整的问题或描述，例如\"谁说过对芒果过敏\"")}},
                             .required = {"query"}},
        });
    }
    if (set.people) {
        tools.push_back({
            .name = "look_up_person",
            .description = "想想她对某个群友知道些什么（以前记下的这个人的事：住哪、在忙什么、喜欢什么）。结果会告诉你，然后你再决定下一步",
            .input_schema = {.properties = {{"target", property("string", "那人发的某条消息的编号，例如 \"#18345\"")}},
                             .required = {"target"}},
        });
    }
    if (set.web) {
        tools.push_back({
            .name = "search_web",
            .description = "上网搜索她不知道或者拿不准的事：新闻、事实、数据、天气、价格。结果会告诉你，然后你再决定下一步",
            .input_schema = {.properties = {{"query",
                                             property("string", "要查什么，写成一句完整的搜索需求，例如\"杭州西湖区今天下雨吗\"")}},
                             .required = {"query"}},
        });
    }
    tools.push_back({
        .name = "stay_silent",
        .description = "她不说话（或者不再说了），这一轮到此为止",
        .input_schema = {.properties = {{"reason", property("string", "简短的理由")}}, .required = {"reason"}},
    });
    return tools;
}

std::expected<Plan, std::string> parse_call(const provider::ToolCall &call, const std::vector<store::StoredMessage> &rows) {
    const auto &input = call.input;
    // A target not shown: let it look again.
    const auto no_target = [&] {
        return std::unexpected(std::string(call.name == "look_up_person" ? "没找到这个编号的消息。" : "没找到这个编号的消息，没有发。"));
    };
    if (call.name == "stay_silent") return Plan{.reason = string_field(input, "reason")};
    if (call.name == "repeat" || call.name == "poke" || call.name == "look_up_person") {
        const auto *target = find_target(string_field(input, "target"), rows);
        if (!target) return no_target();
        return Plan{.target = target->message_id, .repeat = call.name == "repeat", .poke = call.name == "poke"};
    }
    if (call.name == "send_sticker") return Plan{.sticker = string_field(input, "sticker")};
    if (call.name == "wait") {
        // A number, or a string that starts with one ("5秒"). MaiBot waits 30
        // seconds without one; the planner is told instead.
        std::optional<i64> seconds;
        if (const auto *value = field(input, "seconds"); value && value->is_number()) {
            seconds = static_cast<i64>(std::clamp(value->get_number(), 1.0, 60.0));
        } else if (value && value->is_string()) {
            const auto &text = value->get_string();
            if (i64 parsed = 0; std::from_chars(text.data(), text.data() + text.size(), parsed).ec == std::errc{}) seconds = parsed;
        }
        if (!seconds) return std::unexpected(std::string("没说等多少秒，没有等；seconds 填 1 到 60 之间的数字。"));
        return Plan{.wait = std::clamp<i64>(*seconds, 1, 60), .reason = string_field(input, "reason")};
    }
    if (call.name != "reply") return Plan{};

    const auto *target = find_target(string_field(input, "target"), rows);
    if (!target) return no_target();
    const auto style = string_field(input, "style");
    auto to_her = std::string(trim(string_field(input, "to_her")));
    if (to_her != "called" && to_her != "likely" && to_her != "group" && to_her != "unsure" && to_her != "no") to_her.clear();
    return Plan{
        .reply = true,
        .target = target->message_id,
        .to_her = std::move(to_her),
        .quote = bool_field(input, "quote"),
        .style = style == "long"   ? ReplyStyle::LONG :
                 style == "normal" ? ReplyStyle::NORMAL :
                                     ReplyStyle::SHORT,
        .intent = string_field(input, "intent"),
    };
}

Plan parse_plan(const std::vector<provider::OutputItem> &items, const std::vector<store::StoredMessage> &rows) {
    for (const auto &item : items) {
        if (const auto *tool = std::get_if<provider::ToolCallItem>(&item)) {
            auto plan = parse_call(tool->call, rows);
            return plan ? *std::move(plan) : Plan{.reason = std::move(plan).error()};
        }
    }
    return {.reason = "no tool call"};
}

} // namespace lilia::chat
