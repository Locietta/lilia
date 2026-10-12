#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <lilia/store/store.h>

namespace lilia::chat {

// Who each line is said to, from what it says explicitly: an @, a quote, her
// name. Language models find the addressee of a group-chat line far better
// with this structure in view (speakers, reply-to links, a who-to-whom
// summary) than from the text alone; "你" between two friends is the classic
// misreading. Only explicit signals go here; reading the rest is the
// planner's job.

struct LineEdges {
    /// The row it quotes (0: none, or not stored).
    i64 quoted_row = 0;
    /// "[回复 小明: …] " for her own lines, whose stored text lacks the quote.
    std::string quote;
    /// It @s her or quotes her.
    bool to_her = false;
    /// Her name or an alias is in its text (to her, or about her).
    bool names_her = false;
    /// Others it @s or quotes, by name.
    std::vector<std::string> to;
    /// @全体成员.
    bool to_all = false;
};

/// The edges of a stored line, from its segments: `lookup` finds a quoted
/// message by id, `name_of` names an @-ed person whose segment has no name.
LineEdges line_edges(const store::StoredMessage &line, i64 self_id, std::string_view self_name, const std::vector<std::string> &aliases,
                     const std::function<std::optional<store::StoredMessage>(i64)> &lookup, const std::function<std::string(i64)> &name_of);

/// "→她", "→小明、阿紫", "→全体" or empty: how a line shows its addressee.
std::string edge_label(const LineEdges &edges);

/// The text of a line for the planner: "#row" put into its quote ("[回复 #1201
/// 小明: …] "), her own quote restored.
std::string text_with_quote(const store::StoredMessage &line, const LineEdges &edges);

/// "最近谁在对谁说（按 @ 和引用）：小红→小明 3 次（最近 #1250）、她→阿明 1 次（最近 #1248）。" and
/// "阿明和小红在一来一回地聊（#1240–#1252）。" for the newest run of 4+ lines
/// between two people without her; from the rows of the last
/// `window_seconds`; empty when nothing explicit is there.
std::string interaction_summary(const std::vector<store::StoredMessage> &rows, const std::vector<LineEdges> &edges, i64 now,
                                i64 window_seconds);

} // namespace lilia::chat
