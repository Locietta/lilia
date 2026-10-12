#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <lilia/learn/learner.h>
#include <lilia/store/store.h>

namespace lilia::sticker {

// What a sticker is used for, learned from the chats it turns up in (after
// MaiBot's jargon cycle): each sighting keeps an anonymised snippet of the
// lines around it, and at 3, 6, 12, 24, 48 and 96 snippets the helper model
// re-infers the usage from them, with the previous inference as a prior.
// Snippets come from every group the account is in, like jargon evidence:
// speakers are 群友N, names, @mentions, long numbers and links are scrubbed,
// lines are cut short, and only the newest few per sticker are kept.

/// One line around a sighting.
struct ContextLine {
    i64 sender = 0;
    std::string name; ///< the speaker's card or nickname, only for scrubbing
    std::string text;
    i64 time = 0;
    bool self = false; ///< lilia's own line
};

/// The marker standing for the sticker in its snippet.
inline constexpr std::string_view k_sticker_marker = "[这张表情包]";

/// The snippet of a sighting: line `target` is the sticker's (its text, if
/// any, then the marker); speakers 群友N by first appearance, her lines
/// "[SELF] name"; speakers' names inside texts become their labels,
/// "@someone" becomes "@某人", numbers of 5+ digits "[数字]", links "[链接]";
/// every line cut at 80 code points.
std::string usage_snippet(const std::vector<ContextLine> &lines, usize target, std::string_view self_name);

/// Inference is due when a threshold (3, 6, 12, 24, 48, 96) lies in
/// (last, count].
bool usage_due(i64 count, i64 last);

struct Usage {
    std::string usage; ///< "当……时发，表示……"
    std::string tags;  ///< "摇头、拒绝"
    bool no_info = false;
};

/// The inference call (tool record_sticker_usage) over a sticker's picture
/// description, tags, previous usage and snippets from `groups` groups.
learn::LearnerRequest build_usage_request(std::string_view prompt, const store::Sticker &sticker,
                                          const std::vector<store::StickerEvidence> &evidence, usize groups, std::string_view self_name);

/// nullopt when the tool was not called; no_info when the usage is empty.
std::optional<Usage> parse_usage(const std::vector<provider::OutputItem> &items);

} // namespace lilia::sticker
