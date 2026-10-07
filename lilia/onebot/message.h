#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>

#include <lilia/onebot/types.h>

namespace lilia::onebot {

/// Reads a segment data field as text. NapCat sends ids sometimes as JSON
/// strings and sometimes as numbers; numbers come back as integer text.
std::optional<std::string> data_string(const Segment &segment, std::string_view key);

/// Same, parsed as an integer id.
std::optional<i64> data_integer(const Segment &segment, std::string_view key);

Segment text_segment(std::string text);
Segment at_segment(i64 user_id);
Segment reply_segment(i64 message_id);

/// True when the message @-mentions `user_id` (or @all when `include_all`).
bool mentions(const Message &message, i64 user_id, bool include_all = false);

/// Message id quoted by a reply segment, if any.
std::optional<i64> reply_target(const Message &message);

/// Concatenated text segments only.
std::string text_only(const Message &message);

/// Human-readable projection for logs and model context: text verbatim,
/// mentions as "@name", images and stickers as bracketed placeholders.
/// Reply segments are omitted; callers render quoting separately.
std::string render_plain(const Message &message);

/// Same, with descriptions for some segments (by index): an image or sticker
/// with one renders as "[图片: description]" / "[表情: description]".
std::string render_plain(const Message &message, const std::map<usize, std::string> &descriptions);

} // namespace lilia::onebot
