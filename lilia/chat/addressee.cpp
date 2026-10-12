#include "addressee.h"

#include <algorithm>
#include <format>
#include <map>

#include <glaze/json.hpp>

#include <lilia/chat/context.h>
#include <lilia/chat/trigger.h>
#include <lilia/onebot/message.h>

namespace lilia::chat {

namespace {

constexpr std::string_view k_quote_open = "[回复 ";
constexpr usize k_min_dyad_lines = 4;
constexpr usize k_max_pairs = 6;

std::string join(const std::vector<std::string> &names, std::string_view separator) {
    std::string out;
    for (const auto &name : names) {
        if (!out.empty()) out += separator;
        out += name;
    }
    return out;
}

} // namespace

LineEdges line_edges(const store::StoredMessage &line, i64 self_id, std::string_view self_name, const std::vector<std::string> &aliases,
                     const std::function<std::optional<store::StoredMessage>(i64)> &lookup,
                     const std::function<std::string(i64)> &name_of) {
    LineEdges edges;
    onebot::Message segments;
    if (line.segments_json.empty() || glz::read<onebot::k_read_opts>(segments, line.segments_json)) return edges;
    const auto add = [&](std::string name) {
        if (!name.empty() && !std::ranges::contains(edges.to, name)) edges.to.push_back(std::move(name));
    };
    std::string text;
    for (const auto &segment : segments) {
        if (segment.type == "text") {
            text += onebot::data_string(segment, "text").value_or("");
        } else if (segment.type == "reply") {
            const auto id = onebot::reply_target({segment});
            const auto quoted = id ? lookup(*id) : std::nullopt;
            if (!quoted) continue;
            edges.quoted_row = quoted->row_id;
            if (line.from_self) edges.quote = quote_prefix(&*quoted, self_name);
            if (quoted->from_self) {
                if (!line.from_self) edges.to_her = true;
            } else if (quoted->sender_id != line.sender_id || line.from_self) {
                add(display_name(*quoted));
            }
        } else if (segment.type == "at") {
            const auto target = onebot::data_string(segment, "qq").value_or("");
            if (target == "all") {
                edges.to_all = true;
                continue;
            }
            const i64 user = onebot::data_integer(segment, "qq").value_or(0);
            if (user != 0 && user == self_id) {
                if (!line.from_self) edges.to_her = true;
                continue;
            }
            auto name = onebot::data_string(segment, "name").value_or("");
            if (name.starts_with('@')) name.erase(0, 1);
            add(name.empty() ? name_of(user) : name);
        }
    }
    edges.names_her = !line.from_self && contains_alias(text, aliases);
    return edges;
}

std::string edge_label(const LineEdges &edges) {
    std::vector<std::string> names;
    if (edges.to_her) names.emplace_back("她");
    for (const auto &name : edges.to) names.push_back(name);
    if (edges.to_all) names.emplace_back("全体");
    return names.empty() ? std::string() : "→" + join(names, "、");
}

std::string text_with_quote(const store::StoredMessage &line, const LineEdges &edges) {
    std::string text = line.text;
    if (edges.quoted_row <= 0) return text;
    if (!text.starts_with(k_quote_open) && !edges.quote.empty()) text = edges.quote + text;
    if (text.starts_with(k_quote_open)) text.insert(k_quote_open.size(), std::format("#{} ", edges.quoted_row));
    return text;
}

std::string interaction_summary(const std::vector<store::StoredMessage> &rows, const std::vector<LineEdges> &edges, i64 now,
                                i64 window_seconds) {
    struct Pair {
        usize count = 0;
        i64 last_row = 0;
    };
    std::map<std::pair<std::string, std::string>, Pair> pairs;
    std::vector<usize> lines; // people's lines (and hers) in the window
    for (usize i = 0; i < rows.size() && i < edges.size(); ++i) {
        const auto &row = rows[i];
        if (row.time < now - window_seconds || is_notice(row)) continue;
        lines.push_back(i);
        const auto speaker = row.from_self ? std::string("她") : display_name(row);
        std::vector<std::string> targets = edges[i].to;
        if (edges[i].to_her) targets.insert(targets.begin(), "她");
        for (const auto &target : targets) {
            auto &pair = pairs[{speaker, target}];
            ++pair.count;
            pair.last_row = row.row_id;
        }
    }

    // The newest run of 4+ lines between exactly two people, without her.
    std::string dyad;
    usize start = 0;
    for (usize k = 0; k <= lines.size(); ++k) {
        bool extends = false;
        std::vector<i64> speakers;
        if (k < lines.size() && !rows[lines[k]].from_self) {
            for (usize j = start; j <= k; ++j) {
                if (!std::ranges::contains(speakers, rows[lines[j]].sender_id)) speakers.push_back(rows[lines[j]].sender_id);
            }
            extends = speakers.size() <= 2;
        }
        if (extends) continue;
        if (k - start >= k_min_dyad_lines) {
            std::vector<std::string> names;
            for (usize j = start; j < k && names.size() < 2; ++j) {
                if (auto name = display_name(rows[lines[j]]); !std::ranges::contains(names, name)) names.push_back(std::move(name));
            }
            if (names.size() == 2) {
                dyad = std::format("{}和{}在一来一回地聊（#{}–#{}）。", names[0], names[1], rows[lines[start]].row_id,
                                   rows[lines[k - 1]].row_id);
            }
        }
        // Her line ends a run; a third person starts one with the line before.
        if (k < lines.size()) start = rows[lines[k]].from_self ? k + 1 : (k > start ? k - 1 : k);
    }

    std::vector<std::pair<std::pair<std::string, std::string>, Pair>> sorted(pairs.begin(), pairs.end());
    std::ranges::sort(sorted, [](const auto &a, const auto &b) { return a.second.last_row > b.second.last_row; });
    if (sorted.size() > k_max_pairs) sorted.resize(k_max_pairs);
    std::string out;
    for (const auto &[who, pair] : sorted) {
        out += std::format("{}{}→{} {} 次（最近 #{}）", out.empty() ? "最近谁在对谁说（按 @ 和引用）：" : "、", who.first, who.second,
                           pair.count, pair.last_row);
    }
    if (!out.empty()) out += "。";
    if (!dyad.empty()) out += (out.empty() ? "" : "\n") + dyad;
    return out;
}
} // namespace lilia::chat
