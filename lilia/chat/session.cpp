#include "session.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <set>
#include <variant>

namespace lilia::chat {

namespace {

std::string_view trim(std::string_view text) {
    constexpr std::string_view k_space = " \t\r\n";
    const auto first = text.find_first_not_of(k_space);
    if (first == std::string_view::npos) return {};
    const auto last = text.find_last_not_of(k_space);
    return text.substr(first, last - first + 1);
}

const glz::generic *field(const glz::generic &input, std::string_view key) {
    if (!input.is_object()) return nullptr;
    const auto &object = input.get_object();
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

std::string string_field(const glz::generic &input, std::string_view key) {
    const auto *value = field(input, key);
    return value && value->is_string() ? value->get_string() : std::string();
}

bool bool_field(const glz::generic &input, std::string_view key) {
    const auto *value = field(input, key);
    if (!value) return false;
    if (value->is_boolean()) return value->get_boolean();
    if (value->is_string()) return value->get_string() == "true";
    return false;
}

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
    std::string text;
    if (tail.first_new_row > 0) text += std::format("#{} 起是新消息。\n", tail.first_new_row);
    for (const auto &fact : tail.facts) text += fact + "\n";
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

std::string session_line(const store::StoredMessage &message, i32 utc_offset_minutes) {
    return std::format("#{} {}\n", message.row_id, render_line(message, utc_offset_minutes));
}

provider::History render_session(std::string_view prompt, const ChatInfo &info, const std::vector<store::StoredMessage> &rows,
                                 const std::deque<PlannerRecord> &records, const PlannerTail &tail) {
    provider::History history;
    provider::append_system(history, std::string(prompt));

    provider::Item user{.role = provider::Role::USER};
    std::string lines = chat_place(info) + "\n\n聊天记录（行首是消息编号）：\n";
    const auto flush_lines = [&] {
        if (lines.empty()) return;
        user.parts.push_back(provider::TextPart{.text = std::exchange(lines, {})});
    };
    const auto flush_user = [&] {
        flush_lines();
        if (!user.parts.empty()) history.push_back(std::exchange(user, provider::Item{.role = provider::Role::USER}));
    };

    usize next = 0;
    for (const auto &record : records) {
        while (next < rows.size() && rows[next].row_id <= record.after_row) lines += session_line(rows[next++], tail.utc_offset_minutes);
        flush_user();
        for (const auto &item : record.output) provider::append_output_item(history, item);
        // The results first: Anthropic wants them right after the calls.
        for (const auto &outcome : record.outcomes) user.parts.push_back(outcome);
    }
    while (next < rows.size()) lines += session_line(rows[next++], tail.utc_offset_minutes);
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

std::vector<provider::ToolDefinition> planner_tools(const PlannerToolSet &set) {
    const bool reference_handoff = set.reference_handoff;
    std::vector<provider::ToolDefinition> tools;
    tools.push_back({
        .name = "reply",
        .description = "让她说话，回应其中一条消息",
        .input_schema = {.properties = {{"target", property("string", "要回应的消息编号，例如 \"#18345\"")},
                                        {"quote", property("boolean", "是否引用那条消息")},
                                        {"style", property("string", "short | normal | long")},
                                        {reference_handoff ? "brief" : "intent",
                                         reference_handoff ? property("string", "给她的参考信息，不是台词，也不规定她怎么说") :
                                                             property("string", "一句话描述她想表达的意思，不是台词")}},
                         .required = {"target", "style", reference_handoff ? "brief" : "intent"}},
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

Plan parse_plan(const std::vector<provider::OutputItem> &items, const std::vector<store::StoredMessage> &rows) {
    for (const auto &item : items) {
        const auto *tool = std::get_if<provider::ToolCallItem>(&item);
        if (!tool) continue;
        const auto &input = tool->call.input;
        if (tool->call.name == "stay_silent") return {.reason = string_field(input, "reason")};
        if (tool->call.name == "repeat") {
            const auto *target = find_target(string_field(input, "target"), rows);
            if (!target) return {.reason = "invalid repeat target " + string_field(input, "target")};
            return {.target = target->message_id, .repeat = true};
        }
        if (tool->call.name == "poke") {
            const auto *target = find_target(string_field(input, "target"), rows);
            if (!target) return {.reason = "invalid poke target " + string_field(input, "target")};
            return {.target = target->message_id, .poke = true};
        }
        if (tool->call.name == "look_up_person") {
            const auto *target = find_target(string_field(input, "target"), rows);
            if (!target) return {.reason = "invalid look_up_person target " + string_field(input, "target")};
            return {.target = target->message_id, .look_up = true};
        }
        if (tool->call.name == "send_sticker") {
            const auto sticker = string_field(input, "sticker");
            if (sticker.empty()) return {.reason = "send_sticker without a sticker"};
            return {.sticker = sticker};
        }
        if (tool->call.name == "wait") {
            // MaiBot waits 30 seconds when the argument is missing or invalid.
            i64 seconds = 30;
            if (const auto *value = field(input, "seconds"); value && value->is_number()) {
                seconds = static_cast<i64>(value->get_number());
            } else if (value && value->is_string()) {
                const auto &text = value->get_string();
                if (i64 parsed = 0; std::from_chars(text.data(), text.data() + text.size(), parsed).ec == std::errc{}) seconds = parsed;
            }
            return {.wait = std::clamp<i64>(seconds, 1, 60), .reason = string_field(input, "reason")};
        }
        if (tool->call.name != "reply") return {.reason = "unknown tool " + tool->call.name};

        const auto *target = find_target(string_field(input, "target"), rows);
        if (!target) return {.reason = "invalid target " + string_field(input, "target")};
        const auto style = string_field(input, "style");
        return {
            .reply = true,
            .target = target->message_id,
            .quote = bool_field(input, "quote"),
            .style = style == "long"   ? ReplyStyle::LONG :
                     style == "normal" ? ReplyStyle::NORMAL :
                                         ReplyStyle::SHORT,
            .intent = string_field(input, "intent"),
            .brief = string_field(input, "brief"),
        };
    }
    return {.reason = "no tool call"};
}

} // namespace lilia::chat
