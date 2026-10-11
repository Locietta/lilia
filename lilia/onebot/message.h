#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>

#include <glaze/json.hpp>

#include <lilia/onebot/types.h>

namespace lilia::onebot {

/// The text a recalled message is stored with (its segments are kept).
inline constexpr std::string_view k_recalled_text = "[已撤回]";

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
/// mentions as "@name", images and stickers as bracketed placeholders, QQ
/// faces by name, cards by their title. Reply segments are omitted; callers
/// render quoting separately.
std::string render_plain(const Message &message);

/// Same, with descriptions for some segments (by index): an image or sticker
/// with one renders as "[图片: description]" / "[表情: description]", a
/// forward as "[聊天记录: description]".
std::string render_plain(const Message &message, const std::map<usize, std::string> &descriptions);

/// The name of a QQ system face ("微笑" for 14); empty when unknown.
std::string_view face_name(i64 id);

/// "[卡片: 提示｜标题｜描述]" for a json segment's card text; "[卡片]" when
/// nothing readable is in it.
std::string render_card(std::string_view card_json);

/// The inline nodes of a forward segment (NapCat sends them with
/// parseMultMsg); null when it carries only an id.
const glz::generic_i64 *forward_nodes(const Segment &segment);

/// A preview of forwarded nodes, as sent inline or by get_forward_msg
/// ("A: … / B: …（共N条）"): the first few, bounded; empty when there are none.
/// Accepts the node list itself or an object holding it under "messages".
std::string render_forward(const glz::generic_i64 &nodes);

} // namespace lilia::onebot
