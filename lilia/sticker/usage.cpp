#include "usage.h"

#include <algorithm>
#include <array>
#include <format>
#include <unordered_map>

#include <lilia/chat/context.h>
#include <lilia/chat/delivery.h>
#include <lilia/json.h>

namespace lilia::sticker {

namespace {

constexpr usize k_max_line_code_points = 80;
constexpr std::array<i64, 6> k_usage_thresholds{3, 6, 12, 24, 48, 96};

/// The byte length of the first `count` code points of `text`.
usize prefix_bytes(std::string_view text, usize count) {
    usize at = 0;
    for (usize seen = 0; at < text.size() && seen < count; ++seen) {
        ++at;
        while (at < text.size() && (static_cast<u8>(text[at]) & 0xc0) == 0x80) ++at;
    }
    return at;
}

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
bool is_digit(char c) { return c >= '0' && c <= '9'; }

/// Links, typed @mentions and long numbers out of a line.
std::string scrub(std::string_view text) {
    std::string out;
    for (usize i = 0; i < text.size();) {
        const auto rest = text.substr(i);
        if (rest.starts_with("http://") || rest.starts_with("https://")) {
            while (i < text.size() && !is_space(text[i])) ++i;
            out += "[链接]";
        } else if (text[i] == '@') {
            // A typed mention: up to the next space, at most 8 code points.
            usize end = i + 1;
            for (usize seen = 0; end < text.size() && !is_space(text[end]) && seen < 8; ++seen) {
                end += prefix_bytes(text.substr(end), 1);
            }
            out += end > i + 1 ? "@某人" : "@";
            i = end;
        } else if (is_digit(text[i])) {
            usize end = i;
            while (end < text.size() && is_digit(text[end])) ++end;
            out += end - i >= 5 ? std::string("[数字]") : std::string(text.substr(i, end - i));
            i = end;
        } else {
            out += text[i++];
        }
    }
    return out;
}

void replace_all(std::string &text, std::string_view from, std::string_view to) {
    if (from.empty()) return;
    for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) text.replace(at, from.size(), to);
}

} // namespace

std::string usage_snippet(const std::vector<ContextLine> &lines, usize target, std::string_view self_name) {
    std::unordered_map<i64, usize> speakers;
    std::vector<std::string> labels;
    std::vector<std::pair<std::string, std::string>> names; // name -> label, longest first
    for (const auto &line : lines) {
        if (line.self) {
            labels.push_back(std::format("[SELF] {}", self_name));
            continue;
        }
        const auto [it, inserted] = speakers.try_emplace(line.sender, speakers.size() + 1);
        labels.push_back(std::format("群友{}", it->second));
        if (inserted && chat::count_code_points(line.name) >= 2) names.emplace_back(line.name, labels.back());
    }
    std::ranges::sort(names, [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
    std::string snippet;
    for (usize i = 0; i < lines.size(); ++i) {
        auto text = scrub(lines[i].text);
        for (const auto &[name, label] : names) replace_all(text, name, label);
        if (chat::count_code_points(text) > k_max_line_code_points) text = text.substr(0, prefix_bytes(text, k_max_line_code_points)) + "…";
        if (i == target) text = text.empty() ? std::string(k_sticker_marker) : text + " " + std::string(k_sticker_marker);
        if (text.empty()) continue;
        snippet += std::format("{}: {}\n", labels[i], text);
    }
    return snippet;
}

bool usage_due(i64 count, i64 last) {
    return std::ranges::any_of(k_usage_thresholds, [&](i64 threshold) { return last < threshold && threshold <= count; });
}

learn::LearnerRequest build_usage_request(std::string_view prompt, const store::Sticker &sticker,
                                          const std::vector<store::StickerEvidence> &evidence, usize groups, std::string_view self_name) {
    std::string contexts;
    for (usize i = 0; i < evidence.size(); ++i) {
        contexts += std::format("【片段 {}{}】\n{}\n", i + 1, evidence[i].own ? "（她自己发的）" : "", evidence[i].snippet);
    }
    const bool previous = !sticker.usage.empty();
    learn::LearnerRequest request;
    provider::append_user(request.history,
                          chat::fill_template(prompt, {{"description", sticker.description},
                                                       {"tags", sticker.tags},
                                                       {"previous", previous ? "\n上一次总结的用法（仅供参考）：" + sticker.usage : ""},
                                                       {"groups", std::to_string(std::max<usize>(groups, 1))},
                                                       {"name", self_name},
                                                       {"contexts", contexts},
                                                       {"previous_instruction",
                                                        previous ? "\n- 参考上一次的总结，结合新的片段给出更准确的总结；新片段和上一次的"
                                                                   "总结矛盾时，以多数片段为准。" :
                                                                   ""}}));
    provider::SchemaProperty no_info{.type = "boolean", .description = "片段太少或者太杂、看不出固定用法时为 true"};
    request.tools.push_back({
        .name = "record_sticker_usage",
        .description = "提交总结出的表情包用法",
        .input_schema = {.properties = {{"usage", {.type = "string", .description = "一句话：当……时发，表示……（四十字以内）"}},
                                        {"tags", {.type = "string", .description = "最多 5 个标签，用顿号分开"}},
                                        {"no_info", std::move(no_info)}},
                         .required = {"usage", "tags", "no_info"}},
    });
    return request;
}

std::optional<Usage> parse_usage(const std::vector<provider::OutputItem> &items) {
    const auto *input = learn::tool_input(items, "record_sticker_usage");
    if (!input) return std::nullopt;
    Usage usage{.usage = text_field(*input, "usage"), .tags = text_field(*input, "tags"), .no_info = bool_field(*input, "no_info")};
    if (usage.usage.empty()) usage.no_info = true;
    // At most five tags, separated the one way the pool uses.
    replace_all(usage.tags, "，", "、");
    replace_all(usage.tags, ",", "、");
    usize tags = 0;
    for (usize at = 0; at != std::string::npos && at < usage.tags.size();) {
        const auto next = usage.tags.find("、", at);
        if (++tags == 5 && next != std::string::npos) {
            usage.tags.resize(next);
            break;
        }
        at = next == std::string::npos ? next : next + std::string_view("、").size();
    }
    return usage;
}

} // namespace lilia::sticker
