#include "learner.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <format>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include <glaze/json.hpp>

#include <lighter/async/runtime/timeout.h>

#include <xmake/bin2obj/jargon_compare_md.hpp>
#include <xmake/bin2obj/jargon_infer_md.hpp>
#include <xmake/bin2obj/jargon_infer_word_md.hpp>
#include <xmake/bin2obj/jargon_md.hpp>
#include <xmake/bin2obj/learner_md.hpp>
#include <xmake/bin2obj/review_md.hpp>

#include <lilia/chat/context.h>
#include <lilia/chat/delivery.h>
#include <lilia/learn/frequency.h>
#include <lilia/log.h>
#include <lilia/memory/memory.h>
#include <lilia/onebot/message.h>

namespace lilia::learn {

using lighter::Task;

namespace {

constexpr usize k_max_field_code_points = 30;
constexpr usize k_max_line_code_points = 150;
/// More items than this means the model went off the rails: drop the batch.
constexpr usize k_max_learned_items = 20;
constexpr usize k_max_kept_items = 10;
constexpr usize k_max_jargon_items = 30;
constexpr usize k_min_jargon_code_points = 2;
constexpr usize k_max_jargon_code_points = 12;
constexpr usize k_evidence_radius = 3;
constexpr std::array<i64, 4> k_inference_thresholds{4, 8, 25, 100};
constexpr i64 k_complete_count = 100;
constexpr usize k_embed_batch = 16;
constexpr std::chrono::seconds k_call_deadline{90};

std::string_view trim(std::string_view text) {
    constexpr std::string_view k_space = " \t\r\n";
    const auto first = text.find_first_not_of(k_space);
    if (first == std::string_view::npos) return {};
    const auto last = text.find_last_not_of(k_space);
    return text.substr(first, last - first + 1);
}

const glz::generic *field(const glz::generic &object, std::string_view key) {
    if (!object.is_object()) return nullptr;
    const auto &map = object.get_object();
    const auto it = map.find(key);
    return it == map.end() ? nullptr : &it->second;
}

std::string text_field(const glz::generic &object, std::string_view key) {
    const auto *value = field(object, key);
    if (!value) return {};
    if (value->is_string()) return std::string(trim(value->get_string()));
    if (value->is_number()) return std::to_string(static_cast<i64>(value->get_number()));
    return {};
}

bool bool_field(const glz::generic &object, std::string_view key) {
    const auto *value = field(object, key);
    if (!value) return false;
    if (value->is_boolean()) return value->get_boolean();
    return value->is_string() && (value->get_string() == "true" || value->get_string() == "是");
}

/// "3", 3 or "#3" -> 3; 0 when it is not a number.
usize number_value(const glz::generic &value) {
    if (value.is_number()) return static_cast<usize>(value.get_number());
    if (!value.is_string()) return 0;
    std::string_view text = trim(value.get_string());
    if (text.starts_with('#')) text.remove_prefix(1);
    usize number = 0;
    if (std::from_chars(text.data(), text.data() + text.size(), number).ec != std::errc{}) return 0;
    return number;
}

usize number_field(const glz::generic &object, std::string_view key) {
    const auto *value = field(object, key);
    return value ? number_value(*value) : 0;
}

const glz::generic *tool_input(const std::vector<provider::OutputItem> &items, std::string_view tool) {
    for (const auto &item : items) {
        const auto *call = std::get_if<provider::ToolCallItem>(&item);
        if (call && call->call.name == tool) return &call->call.input;
    }
    return nullptr;
}

/// "被问具体内容时" -> "被问具体内容"; the template adds "当…时".
std::string strip_time_suffix(std::string text) {
    for (const std::string_view suffix : {"的时候", "时候", "时"}) {
        if (text.ends_with(suffix) && text.size() > suffix.size()) {
            text.resize(text.size() - suffix.size());
            break;
        }
    }
    return text;
}

/// Insults slip past the prompt now and then; lilia never attacks people, so
/// these are dropped whatever the model says. Insults whatever the target:
constexpr std::array<std::string_view, 14> k_slurs{"煞笔", "傻逼", "傻b",  "傻B",  "sb",   "SB",   "弱智",
                                                   "脑残", "智障", "去死", "nmsl", "你妈", "尼玛", "狗东西"};

/// Fine about things ("这卡有点区", "这道具纯纯垃圾"), not about people:
/// dropped only when the pattern is aimed at a person. (Not "杂鱼": that is
/// playful little-devil teasing.)
constexpr std::array<std::string_view, 11> k_put_downs{"区", "菜", "垃圾", "废物", "肺物", "笨蛋", "笨", "滚", "闭嘴", "艾草", "没用"};

/// A situation about someone (别人/对方/某人...) or a phrase said to "你".
constexpr std::array<std::string_view, 9> k_person_markers{"别人", "对方", "某人", "某个人", "群友", "他人", "人家", "朋友", "骂人"};

/// Situations that describe putting someone else down.
constexpr std::array<std::string_view, 3> k_blocked_situations{"骂人", "让对方", "让人闭嘴"};

bool fits(std::string_view text) { return !text.empty() && chat::count_code_points(text) <= k_max_field_code_points; }

bool contains(std::string_view text, std::string_view word) { return text.find(word) != std::string_view::npos; }

bool is_ascii_word(std::string_view text) {
    return !text.empty() && std::ranges::all_of(text, [](char c) { return static_cast<unsigned char>(c) < 0x80; });
}

bool is_alnum(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

/// `term` occurs in `text`; ASCII terms only as whole words.
bool occurs(std::string_view text, std::string_view term) {
    if (!is_ascii_word(term)) return contains(text, term);
    for (auto at = text.find(term); at != std::string_view::npos; at = text.find(term, at + 1)) {
        const bool left = at == 0 || !is_alnum(text[at - 1]);
        const bool right = at + term.size() >= text.size() || !is_alnum(text[at + term.size()]);
        if (left && right) return true;
    }
    return false;
}

/// Anonymous labels for one batch: 群友N by first appearance, "[SELF] 名字".
std::vector<std::string> speaker_labels(const std::vector<Line> &lines, std::string_view self_name) {
    std::unordered_map<i64, usize> speakers;
    std::vector<std::string> labels;
    for (const auto &line : lines) {
        if (line.self) {
            labels.push_back(std::format("[SELF] {}", self_name));
            continue;
        }
        const auto [it, inserted] = speakers.try_emplace(line.sender, speakers.size() + 1);
        labels.push_back(std::format("群友{}", it->second));
    }
    return labels;
}

provider::ToolDefinition object_list_tool(std::string name, std::string description, std::string list, std::string list_description,
                                          std::string_view item_schema_json, std::vector<std::string> required = {}) {
    glz::generic item_schema;
    std::ignore = glz::read_json(item_schema, item_schema_json);
    provider::SchemaProperty property{.type = "array", .description = std::move(list_description)};
    property.items.emplace(std::move(item_schema));
    if (required.empty()) required.push_back(list);
    return {
        .name = std::move(name),
        .description = std::move(description),
        .input_schema = {.properties = {{std::move(list), std::move(property)}}, .required = std::move(required)},
    };
}

/// A tool whose single argument is a list of numbers (as strings).
provider::ToolDefinition pick_tool(std::string name, std::string description, std::string field, std::string field_description) {
    return object_list_tool(std::move(name), std::move(description), std::move(field), std::move(field_description),
                            R"({"type": "string"})");
}

std::string prompt_of(std::span<const std::byte> embedded, std::string_view name) {
    return chat::fill_template({reinterpret_cast<const char *>(embedded.data()), embedded.size()}, {{"name", name}});
}

/// with_timeout takes lighter's error type; hand the outcome through as a value.
Task<Result<provider::ProviderCallCompletion>> settle(lighter::Task<provider::ProviderCallCompletion, Error> call) {
    auto outcome = co_await std::move(call);
    co_return outcome;
}

/// Spawned tasks return nothing; what a batch learns is already stored.
Task<> learn_batch(Learner *learner, i64 group, std::vector<Line> lines) {
    ++learner->running;
    std::ignore = co_await learner->learn(group, std::move(lines));
    --learner->running;
}

} // namespace

bool blocked_expression(std::string_view situation, std::string_view style) {
    const auto in_either = [&](std::string_view word) { return contains(style, word) || contains(situation, word); };
    if (std::ranges::any_of(k_slurs, in_either)) return true;
    if (std::ranges::any_of(k_blocked_situations, [&](std::string_view word) { return contains(situation, word); })) return true;
    const bool at_person =
        contains(style, "你") || std::ranges::any_of(k_person_markers, [&](std::string_view word) { return contains(situation, word); });
    return at_person && std::ranges::any_of(k_put_downs, in_either);
}

std::string normalize_style(std::string_view style) {
    std::string out;
    for (usize i = 0; i < style.size();) {
        bool replaced = false;
        for (const std::string_view quote : {"「", "」", "“", "”", "『", "』"}) {
            if (style.substr(i).starts_with(quote)) {
                out += '"';
                i += quote.size();
                replaced = true;
                break;
            }
        }
        if (!replaced) out += style[i++];
    }
    // 说"…" / 回"…" / 使用"…" / 用"…" mean the same thing; one form lets repeats count.
    for (const std::string_view verb : {"说\"", "回\"", "使用\""}) {
        if (out.starts_with(verb)) out = "用\"" + out.substr(verb.size());
    }
    return out;
}

std::string render_lines(const std::vector<Line> &lines, std::string_view self_name) {
    const auto labels = speaker_labels(lines, self_name);
    std::string log;
    for (usize i = 0; i < lines.size(); ++i) log += std::format("{}. {}: {}\n", i + 1, labels[i], lines[i].text);
    return log;
}

LearnerRequest build_learner_request(std::string_view prompt, const std::vector<Line> &lines, std::string_view self_name) {
    LearnerRequest request;
    provider::append_system(request.history, std::string(prompt));
    provider::append_user(request.history, render_lines(lines, self_name));
    request.tools.push_back(object_list_tool("record_expressions", "提交总结出的说话方式（一般 3 到 5 条，最多 10 条）", "items",
                                             "总结结果，可以为空",
                                             R"({"type": "object", "properties": {
        "situation": {"type": "string", "description": "场景，不超过 20 个字"},
        "style": {"type": "string", "description": "说法、句式或梗，不超过 20 个字"},
        "source": {"type": "string", "description": "来源行的编号"}},
        "required": ["situation", "style", "source"]})"));
    return request;
}

std::optional<std::vector<Learned>> parse_learned(const std::vector<provider::OutputItem> &items, const std::vector<Line> &lines,
                                                  const std::vector<std::string> &names) {
    std::vector<Learned> learned;
    const auto *input = tool_input(items, "record_expressions");
    if (!input) return learned;
    const auto *list = field(*input, "items");
    if (!list || !list->is_array()) return learned;
    if (list->get_array().size() > k_max_learned_items) return std::nullopt;
    for (const auto &entry : list->get_array()) {
        auto situation = strip_time_suffix(text_field(entry, "situation"));
        auto style = normalize_style(text_field(entry, "style"));
        const auto source = number_field(entry, "source");
        if (source == 0 || source > lines.size() || lines[source - 1].self || lines[source - 1].text.empty()) continue;
        if (!fits(situation) || !fits(style) || blocked_expression(situation, style)) continue;
        const auto mentions_media = [](std::string_view text) {
            return contains(text, "SELF") || contains(text, "[表情") || contains(text, "[图片");
        };
        if (mentions_media(situation) || mentions_media(style)) continue;
        if (std::ranges::any_of(
                names, [&](const std::string &name) { return !name.empty() && (style == name || style == "用\"" + name + "\""); })) {
            continue;
        }
        learned.push_back({.situation = std::move(situation), .style = std::move(style), .source = source - 1});
        if (learned.size() == k_max_kept_items) break;
    }
    return learned;
}

LearnerRequest build_review_request(std::string_view prompt, const std::vector<Learned> &learned) {
    LearnerRequest request;
    provider::append_system(request.history, std::string(prompt));
    std::string list;
    for (usize i = 0; i < learned.size(); ++i) list += std::format("{}. 当{}时，可以{}\n", i + 1, learned[i].situation, learned[i].style);
    provider::append_user(request.history, std::move(list));
    request.tools.push_back(object_list_tool("review_expressions", "提交不适合留下的说法和理由", "reject", "不适合留下的条目，都合适就为空",
                                             R"({"type": "object", "properties": {
        "id": {"type": "string", "description": "编号"},
        "reason": {"type": "string", "description": "不适合的理由"}},
        "required": ["id", "reason"]})"));
    return request;
}

std::optional<std::vector<Rejected>> parse_review(const std::vector<provider::OutputItem> &items, usize count) {
    const auto *input = tool_input(items, "review_expressions");
    if (!input) return std::nullopt;
    std::vector<Rejected> rejected;
    const auto *list = field(*input, "reject");
    if (!list || !list->is_array()) return rejected;
    for (const auto &entry : list->get_array()) {
        // Tolerate a bare number list as well.
        const usize id = entry.is_object() ? number_field(entry, "id") : number_value(entry);
        if (id == 0 || id > count || std::ranges::contains(rejected, id - 1, &Rejected::index)) continue;
        rejected.push_back({.index = id - 1, .reason = entry.is_object() ? text_field(entry, "reason") : std::string()});
    }
    return rejected;
}

LearnerRequest build_jargon_request(std::string_view prompt, const std::vector<Line> &lines, std::string_view self_name) {
    LearnerRequest request;
    provider::append_system(request.history, std::string(prompt));
    provider::append_user(request.history, render_lines(lines, self_name));
    request.tools.push_back(object_list_tool("record_jargon", "提交可能是黑话的候选词条（最多 30 个）", "items", "候选词条，可以为空",
                                             R"({"type": "object", "properties": {
        "content": {"type": "string", "description": "词条原文"},
        "source": {"type": "string", "description": "出现在哪一行（编号）"}},
        "required": ["content", "source"]})"));
    return request;
}

std::vector<Candidate> parse_jargon(const std::vector<provider::OutputItem> &items, const std::vector<Line> &lines,
                                    const std::vector<std::string> &names) {
    std::vector<Candidate> found;
    const auto *input = tool_input(items, "record_jargon");
    if (!input) return found;
    const auto *list = field(*input, "items");
    if (!list || !list->is_array()) return found;
    for (const auto &entry : list->get_array()) {
        auto content = text_field(entry, "content");
        const auto source = number_field(entry, "source");
        const auto length = chat::count_code_points(content);
        if (length < k_min_jargon_code_points || length > k_max_jargon_code_points) continue;
        if (source == 0 || source > lines.size() || !occurs(lines[source - 1].text, content)) continue;
        if (contains(content, "群友") || contains(content, "SELF") || contains(content, "@")) continue;
        if (std::ranges::any_of(names, [&](const std::string &name) { return !name.empty() && contains(content, name); })) continue;
        if (std::ranges::contains(found, content, &Candidate::content)) continue;
        found.push_back({.content = std::move(content), .source = source - 1});
        if (found.size() == k_max_jargon_items) break;
    }
    return found;
}

std::vector<Candidate> known_hits(const std::vector<Line> &lines, const std::vector<std::string> &terms) {
    std::vector<Candidate> hits;
    for (const auto &term : terms) {
        if (chat::count_code_points(term) < k_min_jargon_code_points) continue;
        for (usize i = 0; i < lines.size(); ++i) {
            if (!lines[i].self && occurs(lines[i].text, term)) {
                hits.push_back({.content = term, .source = i});
                break;
            }
        }
    }
    return hits;
}

std::string evidence_snippet(const std::vector<Line> &lines, usize source, std::string_view self_name) {
    const auto labels = speaker_labels(lines, self_name);
    const usize first = source > k_evidence_radius ? source - k_evidence_radius : 0;
    const usize last = std::min(lines.size(), source + k_evidence_radius + 1);
    std::string snippet;
    for (usize i = first; i < last; ++i) snippet += std::format("{}: {}\n", labels[i], lines[i].text);
    return snippet;
}

bool inference_due(i64 count, i64 last_inference_count) {
    return std::ranges::any_of(k_inference_thresholds,
                               [&](i64 threshold) { return last_inference_count < threshold && threshold <= count; });
}

namespace {

provider::ToolDefinition meaning_tool() {
    glz::generic schema;
    std::ignore = glz::read_json(schema, R"({"type": "boolean"})");
    provider::SchemaProperty no_info{.type = "boolean", .description = "信息不足、无法推断时为 true"};
    return {
        .name = "record_meaning",
        .description = "提交推断出的含义",
        .input_schema = {.properties = {{"meaning", {.type = "string", .description = "详细含义说明；无法推断时可以为空"}},
                                        {"no_info", std::move(no_info)}},
                         .required = {"meaning", "no_info"}},
    };
}

} // namespace

LearnerRequest build_inference_request(std::string_view prompt, const store::Jargon &term,
                                       const std::vector<store::JargonEvidence> &evidence, std::string_view self_name) {
    std::string contexts;
    for (usize i = 0; i < evidence.size(); ++i) contexts += std::format("【对话片段 {}】\n{}\n", i + 1, evidence[i].snippet);
    const bool previous = !term.meaning.empty();
    LearnerRequest request;
    provider::append_user(
        request.history,
        chat::fill_template(prompt, {{"content", term.content},
                                     {"contexts", contexts},
                                     {"name", self_name},
                                     {"previous", previous ? "\n**上一次推断的含义（仅供参考）**\n" + term.meaning : ""},
                                     {"previous_instruction",
                                      previous ? "- 请参考上一次推断的含义，结合新的上下文信息，给出更准确或更新的推断结果。" : ""}}));
    request.tools.push_back(meaning_tool());
    return request;
}

LearnerRequest build_word_inference_request(std::string_view prompt, std::string_view content) {
    LearnerRequest request;
    provider::append_user(request.history, chat::fill_template(prompt, {{"content", content}}));
    request.tools.push_back(meaning_tool());
    return request;
}

std::optional<Meaning> parse_meaning(const std::vector<provider::OutputItem> &items) {
    const auto *input = tool_input(items, "record_meaning");
    if (!input) return std::nullopt;
    Meaning meaning{.meaning = text_field(*input, "meaning"), .no_info = bool_field(*input, "no_info")};
    if (meaning.meaning.empty()) meaning.no_info = true;
    return meaning;
}

LearnerRequest build_compare_request(std::string_view prompt, std::string_view first, std::string_view second) {
    LearnerRequest request;
    provider::append_user(request.history, chat::fill_template(prompt, {{"inference1", first}, {"inference2", second}}));
    request.tools.push_back({
        .name = "record_comparison",
        .description = "提交两个推断是否相似",
        .input_schema = {.properties = {{"is_similar", {.type = "boolean", .description = "主要含义是否相同或类似"}},
                                        {"reason", {.type = "string", .description = "判断理由"}}},
                         .required = {"is_similar", "reason"}},
    });
    return request;
}

std::optional<bool> parse_similar(const std::vector<provider::OutputItem> &items) {
    const auto *input = tool_input(items, "record_comparison");
    if (!input) return std::nullopt;
    return bool_field(*input, "is_similar");
}

bool shares(const std::vector<std::vector<i64>> &groups, i64 a, i64 b) {
    if (groups.empty() || b == 0 || a == b) return true;
    return std::ranges::any_of(groups, [&](const std::vector<i64> &group) {
        const bool everyone = std::ranges::contains(group, i64{0});
        return (everyone || std::ranges::contains(group, a)) && (everyone || std::ranges::contains(group, b));
    });
}

std::vector<usize> mmr(const std::vector<std::vector<float>> &vectors, const std::vector<float> &query, usize limit, double lambda) {
    std::vector<double> relevance;
    for (const auto &vector : vectors) relevance.push_back(memory::cosine(vector, query));
    std::vector<usize> chosen;
    std::vector<bool> taken(vectors.size(), false);
    while (chosen.size() < std::min(limit, vectors.size())) {
        usize best = vectors.size();
        double best_score = -1e9;
        for (usize i = 0; i < vectors.size(); ++i) {
            if (taken[i]) continue;
            double redundancy = 0;
            for (const auto j : chosen) redundancy = std::max(redundancy, memory::cosine(vectors[i], vectors[j]));
            const double score = lambda * relevance[i] - (1 - lambda) * redundancy;
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        if (best == vectors.size()) break;
        taken[best] = true;
        chosen.push_back(best);
    }
    return chosen;
}

std::string jargon_reference(const std::vector<store::Jargon> &terms, const std::vector<store::StoredMessage> &log,
                             const std::vector<store::FrequentTerm> &frequent, usize limit) {
    // MaiBot's maisaka/jargon_context_matcher.py: each slang term is placed
    // at the first of the others' messages it occurs in; slang that is also
    // one of the chat's frequent terms ranks above all the rest.
    std::vector<std::string> texts;
    for (const auto &message : log) {
        if (auto text = normalize_match_text(message.text); !message.from_self && !text.empty()) texts.push_back(std::move(text));
    }
    std::vector<std::pair<const store::Jargon *, std::string>> candidates;
    for (const auto &term : terms) {
        if (term.is_jargon != 1 || term.meaning.empty() || chat::count_code_points(term.content) < k_min_jargon_code_points) continue;
        if (auto key = normalize_match_text(term.content); !key.empty()) candidates.emplace_back(&term, std::move(key));
    }
    std::unordered_map<std::string, const store::FrequentTerm *> frequent_by_term;
    for (const auto &term : frequent) frequent_by_term.try_emplace(normalize_match_text(term.term), &term);
    struct Match {
        const store::Jargon *term;
        double score;
        usize first;
        bool frequent;
    };
    std::vector<Match> matches;
    std::unordered_set<std::string> matched;
    for (usize index = 0; index < texts.size(); ++index) {
        for (const auto &[candidate, key] : candidates) {
            const auto &term = *candidate;
            if (matched.contains(key) || !occurs(texts[index], key)) continue;
            const auto hit = frequent_by_term.find(key);
            double bonus = 0;
            if (hit != frequent_by_term.end()) {
                bonus = 1000.0 + static_cast<double>(hit->second->occurrence_count) * 2.0;
                if (hit->second->rank > 0) bonus += std::max(0.0, 100.0 - static_cast<double>(hit->second->rank));
            }
            matches.push_back({.term = &term,
                               .score = static_cast<double>(term.count) + bonus - static_cast<double>(index) * 0.01,
                               .first = index,
                               .frequent = hit != frequent_by_term.end()});
            matched.insert(key);
        }
    }
    if (matches.empty()) return {};
    std::ranges::sort(matches, [](const Match &a, const Match &b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.first != b.first) return a.first < b.first;
        const auto la = chat::count_code_points(a.term->content);
        const auto lb = chat::count_code_points(b.term->content);
        if (la != lb) return la > lb;
        return a.term->content < b.term->content;
    });
    if (matches.size() > limit) matches.resize(limit);
    std::string text = "以下黑话来自当前上下文中其他用户消息的机械匹配，仅作理解聊天语境的参考：\n";
    for (usize i = 0; i < matches.size(); ++i)
        text += std::format("{}. {}：{}{}\n", i + 1, matches[i].term->content, matches[i].term->meaning,
                            matches[i].frequent ? "，同时命中高频词" : "");
    return text;
}

LearnerRequest build_selection_request(std::string_view prompt, const SelectionInput &input) {
    LearnerRequest request;
    provider::append_system(request.history, std::string(prompt));
    std::string user = "最近的聊天：\n";
    for (const auto &line : input.recent) user += line + "\n";
    user += std::format("\n她这次要回应：{}\n", input.target);
    if (!input.intent.empty()) user += std::format("她想表达的意思：{}\n", input.intent);
    user += "\n候选的表达方式：\n";
    for (usize i = 0; i < input.candidates.size(); ++i) user += std::format("{}. {}\n", i + 1, input.candidates[i]);
    provider::append_user(request.history, std::move(user));
    request.tools.push_back(pick_tool("select_expressions", "提交这次可以参考的表达方式编号", "ids",
                                      std::format("选中的编号，0 到 {} 个，可以为空", input.max_picks)));
    return request;
}

std::vector<usize> parse_picks(const std::vector<provider::OutputItem> &items, std::string_view tool, std::string_view field_name,
                               usize count, usize limit) {
    std::vector<usize> picks;
    const auto *input = tool_input(items, tool);
    if (!input) return picks;
    const auto *list = field(*input, field_name);
    if (!list || !list->is_array()) return picks;
    for (const auto &entry : list->get_array()) {
        usize number = 0;
        if (entry.is_number()) {
            number = static_cast<usize>(entry.get_number());
        } else if (entry.is_string()) {
            std::string_view text = trim(entry.get_string());
            if (text.starts_with('#')) text.remove_prefix(1);
            if (std::from_chars(text.data(), text.data() + text.size(), number).ec != std::errc{}) continue;
        } else {
            continue;
        }
        if (number >= 1 && number <= count && !std::ranges::contains(picks, number - 1)) picks.push_back(number - 1);
        if (picks.size() >= limit) break;
    }
    return picks;
}

std::string learnable_text(const onebot::Message &message) {
    std::string text;
    for (const auto &segment : message) {
        if (segment.type == "text") text += onebot::data_string(segment, "text").value_or("");
    }
    auto trimmed = std::string(trim(text));
    if (trimmed.starts_with('/') || chat::count_code_points(trimmed) > k_max_line_code_points) return {};
    return trimmed;
}

Learner::Learner(const LearningConfig &config, std::string name, std::vector<std::string> aliases, store::Store &store,
                 provider::Provider provider, Spawn spawn)
    : now([] { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }),
      config(config), name(name), names(std::move(aliases)), prompt(prompt_of(xmake::learner_md, name)),
      review_prompt(prompt_of(xmake::review_md, name)), jargon_prompt(prompt_of(xmake::jargon_md, name)),
      infer_prompt({reinterpret_cast<const char *>(xmake::jargon_infer_md.data()), xmake::jargon_infer_md.size()}),
      word_prompt({reinterpret_cast<const char *>(xmake::jargon_infer_word_md.data()), xmake::jargon_infer_word_md.size()}),
      compare_prompt({reinterpret_cast<const char *>(xmake::jargon_compare_md.data()), xmake::jargon_compare_md.size()}), store(store),
      provider(std::move(provider)), spawn(std::move(spawn)) {
    names.push_back(this->name);
}

void Learner::observe(const onebot::MessageEvent &message, i64 self_id) {
    if (!config.enabled || !message.is_group() || message.user_id == self_id) return;
    const i64 group = *message.group_id;
    if (std::ranges::contains(config.exclude_groups, group)) return;
    auto text = learnable_text(message.message);
    if (text.empty()) return;
    auto &buffer = buffers[group];
    buffer.push_back({.sender = message.user_id, .text = std::move(text), .time = message.time});
    // A busy group waiting on the interval or a free slot: keep the newest lines.
    const usize cap = std::max<usize>(config.batch_messages, 2) * 3;
    if (buffer.size() > cap) buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(buffer.size() - cap));
    const auto others = static_cast<usize>(std::ranges::count_if(buffer, [](const Line &line) { return !line.self; }));
    if (others < std::max<usize>(config.batch_messages, 2)) return;
    if (now() - last_batch[group] < config.min_interval_seconds || running >= std::max<usize>(config.max_concurrent, 1)) return;
    last_batch[group] = now();
    if (config.daily_token_budget > 0) {
        // Background work has its own budget, so busy groups cannot eat the chats'.
        auto used = store.tokens_since(now() - 86'400, store::TokenScope::BACKGROUND);
        if (used.has_value() && *used >= config.daily_token_budget) {
            log::warn("learning budget used up ({} >= {} tokens in 24 h); group {} waits", *used, config.daily_token_budget, group);
            return;
        }
    }
    spawn(learn_batch(this, group, std::exchange(buffer, {})));
}

void Learner::observe_own(i64 group, std::string text) {
    if (!config.enabled || text.empty() || std::ranges::contains(config.exclude_groups, group)) return;
    buffers[group].push_back({.text = std::move(text), .self = true});
}

namespace {

/// One learning-side model call with a deadline, recorded as chat type "learn".
Task<std::optional<std::vector<provider::OutputItem>>> run_call(Learner *learner, i64 group, const LearnerRequest &request,
                                                                std::string_view purpose) {
    std::vector<provider::OutputItem> items;
    const provider::StreamCallbacks callbacks{
        .on_item_completed = [&items](const provider::OutputItem &item) { items.push_back(item); },
    };
    const i64 started = learner->now();
    auto settled =
        co_await lighter::with_timeout(settle(learner->provider->complete(request.history, request.tools, callbacks)), k_call_deadline);
    auto completion = settled.has_error() ? Result<provider::ProviderCallCompletion>(lighter::outcome_error(
                                                Error::api("timeout", std::format("{} call exceeded its deadline", purpose), "", true))) :
                                            *std::move(settled);

    store::LlmCall call{.time = started, .chat = {.type = "learn", .id = group}, .model = "learner"};
    if (completion.has_error()) {
        call.error = completion.error().message();
        log::warn("{} call failed: {}", purpose, call.error);
    } else {
        if (!completion->model.empty()) call.model = completion->model;
        call.input_tokens = completion->usage.input_tokens;
        call.output_tokens = completion->usage.output_tokens;
        call.cache_read_tokens = completion->usage.cache_read_tokens;
        call.cache_write_tokens = completion->usage.cache_write_tokens;
        call.stop = completion->stop_detail;
        call.request_id = completion->request_id;
    }
    if (auto recorded = learner->store.add_llm_call(call); recorded.has_error())
        log::error("cannot record model call: {}", recorded.error().message());
    if (completion.has_error()) co_return std::nullopt;
    co_return std::move(items);
}

std::string embedding_text(std::string_view situation, std::string_view style) {
    return std::format("情景：{}\n风格：{}", situation, style);
}

} // namespace

Task<std::optional<std::vector<Rejected>>> Learner::review(i64 group, const std::vector<Learned> &learned) {
    if (learned.empty()) co_return std::vector<Rejected>{};
    const auto items = co_await run_call(this, group, build_review_request(review_prompt, learned), "review");
    if (!items) co_return std::nullopt;
    co_return parse_review(*items, learned.size());
}

void Learner::count_frequent_terms(const store::ChatKey &chat, const std::vector<Line> &lines) {
    std::vector<std::string> texts;
    for (const auto &line : lines) {
        if (!line.self) texts.push_back(line.text);
    }
    const auto tallies = count_terms(texts);
    if (tallies.empty()) return;
    if (auto merged = store.merge_frequent_terms(chat, tallies, now(), k_max_frequent_terms); merged.has_error()) {
        log::error("cannot store frequent terms: {}", merged.error().message());
        return;
    }
    log::info("frequent terms: {} counted in group {}", tallies.size(), chat.id);
}

Task<std::vector<Learned>> Learner::learn(i64 group, std::vector<Line> lines) {
    const i64 started = now();
    const store::ChatKey chat{.type = "group", .id = group};
    i64 first = 0, last = 0;
    for (const auto &line : lines) {
        if (line.self || line.time <= 0) continue;
        first = first == 0 ? line.time : std::min(first, line.time);
        last = std::max(last, line.time);
    }
    if (last > 0) {
        if (auto extended = store.extend_learned_span(group, first, last); extended.has_error())
            log::error("learning: {}", extended.error().message());
    }
    if (std::ranges::contains(frequency_chats, group)) count_frequent_terms(chat, lines);
    std::vector<Learned> stored;
    const auto items = co_await run_call(this, group, build_learner_request(prompt, lines, name), "learning");
    if (items) {
        const auto extracted = parse_learned(*items, lines, names);
        if (!extracted) {
            log::warn("learning: more than {} items; batch discarded", k_max_learned_items);
        } else if (!extracted->empty()) {
            // Nothing is stored unless the review passes it; a failed review keeps nothing.
            const auto rejected = co_await review(group, *extracted);
            if (rejected) {
                std::vector<std::pair<i64, std::string>> to_embed;
                for (usize i = 0; i < extracted->size(); ++i) {
                    const auto &pattern = (*extracted)[i];
                    const auto verdict = std::ranges::find(*rejected, i, &Rejected::index);
                    std::ignore = store.add_expression_review({
                        .time = started,
                        .chat = chat,
                        .situation = pattern.situation,
                        .style = pattern.style,
                        .verdict = verdict == rejected->end() ? "kept" : "rejected",
                        .reason = verdict == rejected->end() ? std::string() : verdict->reason,
                    });
                    if (verdict != rejected->end()) continue;
                    auto id = store.learn_expression(chat, pattern.situation, pattern.style, started);
                    if (id.has_error()) {
                        log::error("cannot store expression: {}", id.error().message());
                        continue;
                    }
                    to_embed.emplace_back(*id, embedding_text(pattern.situation, pattern.style));
                    stored.push_back(pattern);
                }
                if (embedder && !to_embed.empty()) {
                    std::vector<std::string> texts;
                    for (const auto &[id, text] : to_embed) texts.push_back(text);
                    auto vectors = co_await embedder->embed(std::move(texts));
                    if (vectors.has_value() && vectors->size() == to_embed.size()) {
                        for (usize i = 0; i < to_embed.size(); ++i) {
                            std::ignore = store.set_expression_vector(to_embed[i].first, embedder->config.embedding_model,
                                                                      memory::encode_vector((*vectors)[i]));
                        }
                    }
                }
            }
            log::info("learned {} expression(s) from {} lines ({} extracted, {} rejected by review)", stored.size(), lines.size(),
                      extracted->size(), extracted->size() - stored.size());
        }
        if (config.max_expressions > 0) {
            if (auto pruned = store.prune_expressions(config.max_expressions); pruned.has_error())
                log::error("cannot prune expressions: {}", pruned.error().message());
        }
    }
    if (config.jargon) co_await learn_jargon(group, std::move(lines));
    co_return stored;
}

Task<> Learner::learn_jargon(i64 group, std::vector<Line> lines) {
    const store::ChatKey chat{.type = "group", .id = group};
    const auto items = co_await run_call(this, group, build_jargon_request(jargon_prompt, lines, name), "jargon");
    std::vector<Candidate> found;
    if (items) found = parse_jargon(*items, lines, names);
    // Known terms count even when the model misses them (MaiBot's cache hits);
    // confirmed ordinary words stop being counted.
    std::vector<std::string> known;
    if (auto terms = store.jargon_terms(); terms.has_value()) {
        for (const auto &term : *terms) {
            if (term.is_jargon != 0) known.push_back(term.content);
        }
    }
    for (auto &hit : known_hits(lines, known)) {
        if (!std::ranges::contains(found, hit.content, &Candidate::content)) found.push_back(std::move(hit));
    }
    std::vector<store::Jargon> due;
    for (const auto &candidate : found) {
        auto term =
            store.add_jargon_hit(candidate.content, chat, evidence_snippet(lines, candidate.source, name), now(), config.jargon_evidence);
        if (term.has_error()) {
            log::error("cannot store jargon: {}", term.error().message());
            continue;
        }
        if (term->created_by == "AI" && !term->complete && inference_due(term->count, term->last_inference_count)) due.push_back(*term);
    }
    if (!found.empty()) log::info("jargon: {} term(s) seen in group {}, {} due for inference", found.size(), group, due.size());
    for (auto &term : due) co_await infer(std::move(term));
}

Task<> Learner::infer(store::Jargon term) {
    auto evidence = store.jargon_evidence(term.id, config.jargon_evidence);
    if (evidence.has_error() || evidence->empty()) co_return;
    const auto first_items = co_await run_call(this, 0, build_inference_request(infer_prompt, term, *evidence, name), "jargon inference");
    if (!first_items) co_return;
    const auto first = parse_meaning(*first_items);
    if (!first) co_return;
    const bool complete = term.count >= k_complete_count;
    if (first->no_info) {
        // Not enough to go on yet; wait for the next threshold.
        std::ignore = store.set_jargon_inference(term.id, term.meaning, term.is_jargon, term.count, complete);
        log::info("jargon: not enough context for \"{}\" yet", term.content);
        co_return;
    }
    const auto word_items = co_await run_call(this, 0, build_word_inference_request(word_prompt, term.content), "jargon word inference");
    if (!word_items) co_return;
    const auto alone = parse_meaning(*word_items);
    if (!alone) co_return;
    const auto compare_items =
        co_await run_call(this, 0, build_compare_request(compare_prompt, first->meaning, alone->meaning), "jargon compare");
    if (!compare_items) co_return;
    const auto similar = parse_similar(*compare_items);
    if (!similar) co_return;
    // Meaning the same without context means it is an ordinary word.
    const bool jargon = !*similar;
    std::ignore = store.set_jargon_inference(term.id, jargon ? first->meaning : term.meaning, jargon ? 1 : 0, term.count, complete);
    log::info("jargon: \"{}\" is {} ({} sightings){}", term.content, jargon ? "slang" : "an ordinary word", term.count,
              jargon ? ": " + first->meaning : "");
}

Task<> Learner::embed_missing() {
    if (!embedder) co_return;
    auto rows = store.expressions(0);
    if (rows.has_error()) co_return;
    std::vector<store::Expression> missing;
    for (auto &row : *rows) {
        if (row.vector.empty() || row.model != embedder->config.embedding_model) missing.push_back(std::move(row));
    }
    usize done = 0;
    for (usize start = 0; start < missing.size(); start += k_embed_batch) {
        const usize end = std::min(missing.size(), start + k_embed_batch);
        std::vector<std::string> texts;
        for (usize i = start; i < end; ++i) texts.push_back(embedding_text(missing[i].situation, missing[i].style));
        auto vectors = co_await embedder->embed(std::move(texts));
        if (vectors.has_error() || vectors->size() != end - start) break;
        for (usize i = start; i < end; ++i) {
            std::ignore =
                store.set_expression_vector(missing[i].id, embedder->config.embedding_model, memory::encode_vector((*vectors)[i - start]));
            ++done;
        }
    }
    if (done > 0) log::info("learning: embedded {} expression(s)", done);
}

} // namespace lilia::learn
