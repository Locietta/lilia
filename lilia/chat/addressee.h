#pragma once

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <lilia/store/store.h>

namespace lilia::chat {

// Who each line is said to, from what it says explicitly: an @, a quote, her
// name. Language models find the addressee of a group-chat line far better
// with this structure in view (speakers, reply-to links) than from the text
// alone; "你" between two friends is the classic misreading. Only explicit
// signals go here; for lines that name no one, the planner reads them with
// one derived estimate, the addressee prior over line_features (feedback/).

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
    /// How: an @ or a quote, of her or of someone else.
    bool at_her = false;
    bool quotes_her = false;
    bool at_other = false;
    bool quotes_other = false;
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

/// The features of line `index` for the addressee model (feedback/), from
/// the log up to it alone, so live hints, backfills and training agree:
/// bias; @, quote or name of her; @, quote or name of someone else; 你 (not
/// 你们); 你们/大家/谁 or @all; a question; a reaction only; right after her
/// (3 min); someone she was just talking with; her share of the last 5
/// minutes; time since her last line (log-scaled, 1 = never or a day); a
/// run between this speaker and one other without her; other speakers in
/// the last 5 minutes (of 6).
inline constexpr usize k_features = 17;
inline constexpr i64 k_feature_version = 1;
using Features = std::array<double, k_features>;
Features line_features(const std::vector<store::StoredMessage> &rows, const std::vector<LineEdges> &edges, usize index);

} // namespace lilia::chat
