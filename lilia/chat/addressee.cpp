#include "addressee.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>

#include <glaze/json.hpp>

#include <lilia/chat/context.h>
#include <lilia/chat/delivery.h>
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
                if (!line.from_self) edges.to_her = edges.quotes_her = true;
            } else if (quoted->sender_id != line.sender_id || line.from_self) {
                add(display_name(*quoted));
                edges.quotes_other = true;
            }
        } else if (segment.type == "at") {
            const auto target = onebot::data_string(segment, "qq").value_or("");
            if (target == "all") {
                edges.to_all = true;
                continue;
            }
            const i64 user = onebot::data_integer(segment, "qq").value_or(0);
            if (user != 0 && user == self_id) {
                if (!line.from_self) edges.to_her = edges.at_her = true;
                continue;
            }
            edges.at_other = true;
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
Features line_features(const std::vector<store::StoredMessage> &rows, const std::vector<LineEdges> &edges, usize index) {
    constexpr i64 k_after_her_seconds = 180;
    constexpr i64 k_partner_seconds = 600;
    constexpr i64 k_partner_gap_seconds = 60;
    constexpr i64 k_recent_seconds = 300;
    constexpr usize k_dyad_lines = 4;
    constexpr double k_max_speakers = 6;
    Features x{};
    x[0] = 1;
    if (index >= rows.size()) return x;
    const auto &line = rows[index];
    const LineEdges none;
    const auto edges_of = [&](usize at) -> const LineEdges & { return at < edges.size() ? edges[at] : none; };
    const auto &own = edges_of(index);
    const std::string text(without_quote(line.text));
    const auto has = [&](std::string_view word) { return text.find(word) != std::string::npos; };
    const auto count = [&](std::string_view word) {
        usize found = 0;
        for (auto at = text.find(word); at != std::string::npos; at = text.find(word, at + word.size())) ++found;
        return found;
    };

    x[1] = own.at_her;
    x[2] = own.quotes_her;
    x[3] = own.names_her;
    x[4] = own.at_other;
    x[5] = own.quotes_other;
    std::vector<std::string> others;
    for (usize j = 0; j < index; ++j) {
        const auto &row = rows[j];
        if (row.from_self || is_notice(row) || row.sender_id == line.sender_id || line.time - row.time > k_partner_seconds) continue;
        if (auto name = display_name(row); count_code_points(name) >= 2 && !std::ranges::contains(others, name)) others.push_back(name);
    }
    x[6] = std::ranges::any_of(others, [&](const std::string &name) { return has(name); });
    x[7] = count("你") > count("你们") || has("您") || has("妳");
    x[8] = has("你们") || has("大家") || has("各位") || has("有没有人") || has("有人") || has("谁") || own.to_all;
    x[9] = has("？") || has("?") || text.ends_with("吗") || text.ends_with("呢") || text.ends_with("么");
    x[10] = count_code_points(text) <= 4 || text.starts_with("[表情") || text.starts_with("[图片");

    // The line before (not a notice), and her latest line.
    std::optional<usize> previous;
    std::optional<usize> hers;
    for (usize j = index; j-- > 0;) {
        if (is_notice(rows[j])) continue;
        if (!previous) previous = j;
        if (rows[j].from_self) {
            hers = j;
            break;
        }
    }
    x[11] = previous && rows[*previous].from_self && line.time - rows[*previous].time <= k_after_her_seconds;
    if (hers && line.time - rows[*hers].time <= k_partner_seconds) {
        // She answered this speaker, or quoted or @-ed them.
        bool partner = std::ranges::contains(edges_of(*hers).to, display_name(line));
        for (usize j = *hers; !partner && j-- > 0;) {
            if (is_notice(rows[j])) continue;
            partner = rows[j].sender_id == line.sender_id && rows[*hers].time - rows[j].time <= k_partner_gap_seconds;
            break;
        }
        x[12] = partner;
    }
    usize recent = 0;
    usize recent_hers = 0;
    std::vector<i64> speakers;
    for (usize j = 0; j < index; ++j) {
        const auto &row = rows[j];
        if (is_notice(row) || line.time - row.time > k_recent_seconds) continue;
        ++recent;
        if (row.from_self) {
            ++recent_hers;
        } else if (row.sender_id != line.sender_id && !std::ranges::contains(speakers, row.sender_id)) {
            speakers.push_back(row.sender_id);
        }
    }
    x[13] = recent > 0 ? static_cast<double>(recent_hers) / static_cast<double>(recent) : 0;
    x[14] =
        hers ? std::min(1.0, std::log1p(static_cast<double>(std::max<i64>(0, line.time - rows[*hers].time))) / std::log1p(86'400.0)) : 1;
    // The last lines (this one included) between this speaker and one other.
    std::vector<i64> people;
    usize lines = 0;
    bool with_her = false;
    for (usize j = index + 1; j-- > 0 && lines < k_dyad_lines;) {
        if (is_notice(rows[j])) continue;
        ++lines;
        if (rows[j].from_self) with_her = true;
        if (!std::ranges::contains(people, rows[j].sender_id)) people.push_back(rows[j].sender_id);
    }
    x[15] = lines == k_dyad_lines && !with_her && people.size() == 2;
    x[16] = std::min(k_max_speakers, static_cast<double>(speakers.size())) / k_max_speakers;
    return x;
}

} // namespace lilia::chat
